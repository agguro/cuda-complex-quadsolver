# -----------------------------------------------------------------------------
# File: quadratic_solver.s
#
# Purpose:
#   x86_64 Linux host orchestrator for the CUDA quadratic solver.
#
# Responsibilities:
#   1. Parse command line
#   2. Open CSV input
#   3. Map CSV into memory
#   4. Parse coefficients
#   5. Initialize CUDA Driver API
#   6. Load embedded CUBIN
#   7. Allocate GPU memory
#   8. Copy input -> GPU
#   9. Launch CUDA kernel
#  10. Synchronize and check errors
#  11. Copy GPU results -> host
#  12. Print results
#  13. Clean up resources
#
# Input format:
#   a,b,c,d,e,f
#
# The first three values are assumed to be the quadratic coefficients:
#       ax² + bx + c = 0
#
# The remaining three values are preserved in the input row.
#
# Output row:
#   8 doubles:
#       root1_real
#       root1_imag
#       root2_real
#       root2_imag
#       + 4 reserved/output values
#
# -----------------------------------------------------------------------------

.equ DOUBLE_SIZE,       8
.equ PADDED_COEFFS,     8
.equ SIZEOF_INPUT_ROW,  (PADDED_COEFFS * DOUBLE_SIZE)
.equ SIZEOF_OUTPUT_ROW, (PADDED_COEFFS * DOUBLE_SIZE)

.equ MAX_ROWS,          1024

.equ HOST_IN_BUF_SIZE,  (MAX_ROWS * SIZEOF_INPUT_ROW)
.equ HOST_OUT_BUF_SIZE, (MAX_ROWS * SIZEOF_OUTPUT_ROW)

# -----------------------------------------------------------------------------
# Linux syscalls
# -----------------------------------------------------------------------------

.equ SYS_READ,          0
.equ SYS_WRITE,         1
.equ SYS_OPEN,          2
.equ SYS_CLOSE,         3
.equ SYS_MMAP,          9
.equ SYS_MUNMAP,        11
.equ SYS_EXIT_GROUP,    231

.equ O_RDONLY,          0
.equ O_WRONLY,          1
.equ O_CREAT,           64
.equ O_TRUNC,           512

.equ PROT_READ,         1
.equ MAP_PRIVATE,       2

.equ FILE_MODE,         438             # 0666

# -----------------------------------------------------------------------------
# CUDA constants
# -----------------------------------------------------------------------------

.equ CUDA_SUCCESS,      0

# -----------------------------------------------------------------------------
# Strings
# -----------------------------------------------------------------------------

.section .rodata

    .align 16

kernel_bin:
    .incbin "quadratic_solver.cubin"

kernel_name:
    .asciz "quadratic_solver"

usage_msg:
    .asciz \
"Usage: %s <input.csv> [-o <output.csv>]\n"

err_no_input:
    .asciz \
"ERROR: no input CSV specified.\n"

err_open_input:
    .asciz \
"ERROR: could not open input CSV.\n"

err_mmap:
    .asciz \
"ERROR: mmap() failed.\n"

err_malloc:
    .asciz \
"ERROR: malloc() failed.\n"

err_empty:
    .asciz \
"ERROR: input contains no valid rows.\n"

err_too_many:
    .asciz \
"ERROR: input contains more than %ld rows.\n"

err_parse:
    .asciz \
"ERROR: malformed CSV row %ld.\n"

err_cuda:
    .asciz \
"ERROR: CUDA operation failed: %s\n"

err_cuda_code:
    .asciz \
"       CUDA error code: %ld\n"

err_cuda_name:
    .asciz \
"       CUDA error name: %s\n"

err_cuda_string:
    .asciz \
"       CUDA error: %s\n"

err_launch:
    .asciz \
"ERROR: CUDA kernel launch failed.\n"

err_sync:
    .asciz \
"ERROR: CUDA synchronization failed.\n"

err_copy:
    .asciz \
"ERROR: CUDA memory copy failed.\n"

err_alloc:
    .asciz \
"ERROR: CUDA device memory allocation failed.\n"

err_module:
    .asciz \
"ERROR: CUDA module loading failed.\n"

err_function:
    .asciz \
"ERROR: CUDA kernel function lookup failed.\n"

err_init:
    .asciz \
"ERROR: CUDA initialization failed.\n"

err_device:
    .asciz \
"ERROR: CUDA device selection failed.\n"

err_context:
    .asciz \
"ERROR: CUDA context creation failed.\n"

hdr_msg:
    .asciz \
"\n--- GPU Execution Results (Double Precision) ---\n"

res_fmt:
    .asciz \
"Row %ld: Roots -> R1: (%+.8f, %+.8fi) | R2: (%+.8f, %+.8fi)\n"

csv_row_fmt:
    .asciz \
"%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n"

csv_format:
    .asciz \
"%lf,%lf,%lf,%lf,%lf,%lf"

opt_o:
    .asciz "-o"

str_input:
    .asciz "input CSV"

str_cuda_init:
    .asciz "cuInit"

str_cuda_device:
    .asciz "cuDeviceGet"

str_cuda_context:
    .asciz "cuCtxCreate_v2"

str_cuda_module:
    .asciz "cuModuleLoadData"

str_cuda_function:
    .asciz "cuModuleGetFunction"

str_cuda_alloc_input:
    .asciz "cuMemAlloc_v2(input)"

str_cuda_alloc_output:
    .asciz "cuMemAlloc_v2(output)"

str_cuda_copy_htod:
    .asciz "cuMemcpyHtoD_v2"

str_cuda_launch:
    .asciz "cuLaunchKernel"

str_cuda_sync:
    .asciz "cuCtxSynchronize"

str_cuda_copy_dtoh:
    .asciz "cuMemcpyDtoH_v2"

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

.section .data

    .align 16

d_in_ptr:
    .quad 0

d_out_ptr:
    .quad 0

h_module:
    .quad 0

h_func:
    .quad 0

h_context:
    .quad 0

host_input_ptr:
    .quad 0

host_output_ptr:
    .quad 0

mapped_ptr:
    .quad 0

mapped_size:
    .quad 0

input_fd:
    .quad -1

out_fd:
    .quad 1

row_count:
    .quad 0

input_filename:
    .quad 0

output_filename:
    .quad 0

# -----------------------------------------------------------------------------
# Kernel parameters
#
# CUDA expects:
#
#   void *kernelParams[]
#
# where each entry points to the actual argument value.
#
# Kernel:
#
#   quadratic_solver(
#       double *input,
#       double *output,
#       unsigned long N
#   )
# -----------------------------------------------------------------------------

kparam_in_ptr:
    .quad 0

kparam_out_ptr:
    .quad 0

kparam_N:
    .quad 0

k_params:
    .quad kparam_in_ptr
    .quad kparam_out_ptr
    .quad kparam_N
    .quad 0

# -----------------------------------------------------------------------------
# CUDA error storage
# -----------------------------------------------------------------------------

cuda_error_code:
    .quad 0

cuda_error_name_ptr:
    .quad 0

cuda_error_string_ptr:
    .quad 0

# -----------------------------------------------------------------------------
# Text
# -----------------------------------------------------------------------------

.section .text

.globl _start
.type _start,@function

_start:

# =============================================================================
# 1. INITIAL STACK / ARGUMENT VALIDATION
# =============================================================================

    movq    (%rsp), %r15                  # argc
    movq    8(%rsp), %r11                 # argv[0]

    cmpq    $2, %r15
    jl      .L_usage

    movq    16(%rsp), %r12                # argv[1]
    movq    %r12, input_filename(%rip)

    # -------------------------------------------------------------------------
    # Establish ABI-compliant stack before calling libc.
    # -------------------------------------------------------------------------

    pushq   %rbp
    movq    %rsp, %rbp
    andq    $-16, %rsp
    subq    $256, %rsp

    # Default output = stdout
    movq    $1, out_fd(%rip)

# =============================================================================
# 2. PARSE OPTIONAL -o
# =============================================================================

    movq    $2, %r12

.L_arg_loop:

    cmpq    %r15, %r12
    jge     .L_args_done

    movq    8(%rbp,%r12,8), %rdi
    leaq    opt_o(%rip), %rsi
    call    strcmp_local

    testl   %eax, %eax
    jnz     .L_arg_next

    # -o found
    incq    %r12

    cmpq    %r15, %r12
    jge     .L_usage_after_frame

    movq    8(%rbp,%r12,8), %rdi
    movq    %rdi, output_filename(%rip)

    # open(output, O_WRONLY|O_CREAT|O_TRUNC, 0666)
    movq    $SYS_OPEN, %rax
    movq    %rdi, %rdi
    movq    $(O_WRONLY|O_CREAT|O_TRUNC), %rsi
    movq    $FILE_MODE, %rdx
    syscall

    testq   %rax, %rax
    js      .L_error_open_output

    movq    %rax, out_fd(%rip)

.L_arg_next:

    incq    %r12
    jmp     .L_arg_loop

.L_args_done:

# =============================================================================
# 3. OPEN INPUT
# =============================================================================

    movq    input_filename(%rip), %rdi

    movq    $SYS_OPEN, %rax
    movq    $O_RDONLY, %rsi
    xorq    %rdx, %rdx
    syscall

    testq   %rax, %rax
    js      .L_error_open_input

    movq    %rax, input_fd(%rip)

# =============================================================================
# 4. MMAP INPUT
#
# We map 64 KiB for now.
# =============================================================================

    xorq    %rdi, %rdi
    movq    $65536, %rsi
    movq    $PROT_READ, %rdx
    movq    $MAP_PRIVATE, %r10
    movq    input_fd(%rip), %r8
    xorq    %r9, %r9
    movq    $SYS_MMAP, %rax
    syscall

    testq   %rax, %rax
    js      .L_error_mmap

    movq    %rax, mapped_ptr(%rip)
    movq    $65536, mapped_size(%rip)

# =============================================================================
# 5. HOST ALLOCATION
# =============================================================================

    movq    $HOST_IN_BUF_SIZE, %rdi
    call    malloc@PLT

    testq   %rax, %rax
    jz      .L_error_malloc

    movq    %rax, host_input_ptr(%rip)

    movq    $HOST_OUT_BUF_SIZE, %rdi
    call    malloc@PLT

    testq   %rax, %rax
    jz      .L_error_malloc

    movq    %rax, host_output_ptr(%rip)

# =============================================================================
# 6. PARSE CSV
# =============================================================================

    movq    mapped_ptr(%rip), %r12
    xorq    %rbx, %rbx

.L_scan_loop:

    cmpq    $MAX_ROWS, %rbx
    jge     .L_error_too_many

    # destination = host_input + row * 64

    movq    %rbx, %rax
    imulq   $SIZEOF_INPUT_ROW, %rax
    addq    host_input_ptr(%rip), %rax

    # sscanf(
    #   source,
    #   format,
    #   &a,&b,&c,&d,&e,&f
    # )

    movq    %r12, %rdi
    leaq    csv_format(%rip), %rsi

    movq    %rax, %rdx
    leaq    8(%rax), %rcx
    leaq    16(%rax), %r8
    leaq    24(%rax), %r9

    # arguments 6 and 7 go on stack
    subq    $16, %rsp

    leaq    32(%rax), %r10
    movq    %r10, 0(%rsp)

    leaq    40(%rax), %r10
    movq    %r10, 8(%rsp)

    xorl    %eax, %eax
    call    sscanf@PLT

    addq    $16, %rsp

    # Must have parsed six doubles.
    cmpq    $6, %rax
    jne     .L_parse_error

    # -------------------------------------------------------------------------
    # Find end of current line.
    # -------------------------------------------------------------------------

.L_find_newline:

    movb    (%r12), %al

    cmpb    $10, %al
    je      .L_next_row

    cmpb    $0, %al
    je      .L_last_row

    incq    %r12
    jmp     .L_find_newline

.L_next_row:

    incq    %r12
    incq    %rbx
    jmp     .L_scan_loop

.L_last_row:

    incq    %rbx

# =============================================================================
# 7. VALIDATE ROW COUNT
# =============================================================================

.L_scan_done:

    testq   %rbx, %rbx
    jz      .L_error_empty

    movq    %rbx, row_count(%rip)

# =============================================================================
# 8. CUDA INITIALIZATION
# =============================================================================

    xorl    %edi, %edi
    call    cuInit@PLT

    testl   %eax, %eax
    jnz     .L_cuda_init_error

# =============================================================================
# 9. GET DEVICE
# =============================================================================

    leaq    128(%rsp), %rdi
    xorl    %esi, %esi

    call    cuDeviceGet@PLT

    testl   %eax, %eax
    jnz     .L_cuda_device_error

# =============================================================================
# 10. CREATE CUDA CONTEXT
# =============================================================================

    leaq    136(%rsp), %rdi
    xorl    %esi, %esi
    movl    128(%rsp), %edx

    call    cuCtxCreate_v2@PLT

    testl   %eax, %eax
    jnz     .L_cuda_context_error

    movq    136(%rsp), %rax
    movq    %rax, h_context(%rip)

# =============================================================================
# 11. LOAD CUBIN
# =============================================================================

    leaq    h_module(%rip), %rdi
    leaq    kernel_bin(%rip), %rsi

    call    cuModuleLoadData@PLT

    testl   %eax, %eax
    jnz     .L_cuda_module_error

# =============================================================================
# 12. GET KERNEL FUNCTION
# =============================================================================

    leaq    h_func(%rip), %rdi
    movq    h_module(%rip), %rsi
    leaq    kernel_name(%rip), %rdx

    call    cuModuleGetFunction@PLT

    testl   %eax, %eax
    jnz     .L_cuda_function_error

# =============================================================================
# 13. GPU MEMORY ALLOCATION
# =============================================================================

    leaq    d_in_ptr(%rip), %rdi
    movq    row_count(%rip), %rsi
    imulq   $SIZEOF_INPUT_ROW, %rsi

    call    cuMemAlloc_v2@PLT

    testl   %eax, %eax
    jnz     .L_cuda_alloc_error

    leaq    d_out_ptr(%rip), %rdi
    movq    row_count(%rip), %rsi
    imulq   $SIZEOF_OUTPUT_ROW, %rsi

    call    cuMemAlloc_v2@PLT

    testl   %eax, %eax
    jnz     .L_cuda_alloc_error

# =============================================================================
# 14. COPY HOST -> DEVICE
# =============================================================================

    movq    d_in_ptr(%rip), %rdi
    movq    host_input_ptr(%rip), %rsi

    movq    row_count(%rip), %rdx
    imulq   $SIZEOF_INPUT_ROW, %rdx

    call    cuMemcpyHtoD_v2@PLT

    testl   %eax, %eax
    jnz     .L_cuda_copy_error

# =============================================================================
# 15. PREPARE KERNEL PARAMETERS
# =============================================================================

    movq    d_in_ptr(%rip), %rax
    movq    %rax, kparam_in_ptr(%rip)

    movq    d_out_ptr(%rip), %rax
    movq    %rax, kparam_out_ptr(%rip)

    movq    row_count(%rip), %rax
    movq    %rax, kparam_N(%rip)

# =============================================================================
# 16. KERNEL LAUNCH
#
# We use:
#
#       grid  = N blocks
#       block = 1 thread
#
# This is deliberately simple.
#
# Each block processes one row.
# =============================================================================

    movq    h_func(%rip), %rdi

    # gridDimX
    movl    row_count(%rip), %esi

    # gridDimY
    movl    $1, %edx

    # gridDimZ
    movl    $1, %ecx

    # blockDimX
    movl    $1, %r8d

    # blockDimY
    movl    $1, %r9d

    # Stack:
    #
    # 7th  = blockDimZ
    # 8th  = sharedMemBytes
    # 9th  = stream
    # 10th = kernelParams
    # 11th = extra
    #
    subq    $40, %rsp

    movq    $1, 0(%rsp)                       # blockDimZ
    movq    $0, 8(%rsp)                       # shared memory
    movq    $0, 16(%rsp)                      # default stream

    leaq    k_params(%rip), %rax
    movq    %rax, 24(%rsp)                    # kernelParams

    movq    $0, 32(%rsp)                      # extra

    call    cuLaunchKernel@PLT

    addq    $40, %rsp

    testl   %eax, %eax
    jnz     .L_cuda_launch_error

# =============================================================================
# 17. SYNCHRONIZE
#
# This is important because a kernel launch can be asynchronous.
# =============================================================================

    call    cuCtxSynchronize@PLT

    testl   %eax, %eax
    jnz     .L_cuda_sync_error

# =============================================================================
# 18. COPY DEVICE -> HOST
# =============================================================================

    movq    host_output_ptr(%rip), %rdi
    movq    d_out_ptr(%rip), %rsi

    movq    row_count(%rip), %rdx
    imulq   $SIZEOF_OUTPUT_ROW, %rdx

    call    cuMemcpyDtoH_v2@PLT

    testl   %eax, %eax
    jnz     .L_cuda_copy_error

# =============================================================================
# 19. PRINT RESULTS
# =============================================================================

    movq    out_fd(%rip), %r15

    cmpq    $1, %r15
    jne     .L_print_loop

    leaq    hdr_msg(%rip), %rdi
    xorl    %eax, %eax
    call    printf@PLT

.L_print_loop:

    xorq    %r12, %r12

.L_print_loop_body:

    cmpq    row_count(%rip), %r12
    jge     .L_cleanup

    # input row pointer
    movq    %r12, %rax
    imulq   $SIZEOF_INPUT_ROW, %rax
    addq    host_input_ptr(%rip), %rax
    movq    %rax, %rcx

    # output row pointer
    movq    %r12, %rax
    imulq   $SIZEOF_OUTPUT_ROW, %rax
    addq    host_output_ptr(%rip), %rax
    movq    %rax, %rdx

    cmpq    $1, %r15
    je      .L_terminal_output

# -----------------------------------------------------------------------------
# CSV OUTPUT
# -----------------------------------------------------------------------------

    movsd   0(%rcx), %xmm0
    movsd   8(%rcx), %xmm1
    movsd   16(%rcx), %xmm2
    movsd   24(%rcx), %xmm3
    movsd   32(%rcx), %xmm4
    movsd   40(%rcx), %xmm5

    movsd   0(%rdx), %xmm6
    movsd   8(%rdx), %xmm7
    movsd   16(%rdx), %xmm8
    movsd   24(%rdx), %xmm9

    subq    $16, %rsp

    movsd   %xmm8, 0(%rsp)
    movsd   %xmm9, 8(%rsp)

    movq    %r15, %rdi
    leaq    csv_row_fmt(%rip), %rsi

    # 8 floating-point arguments
    movl    $8, %eax

    call    dprintf@PLT

    addq    $16, %rsp

    jmp     .L_loop_inc

# -----------------------------------------------------------------------------
# TERMINAL OUTPUT
# -----------------------------------------------------------------------------

.L_terminal_output:

    leaq    res_fmt(%rip), %rdi

    movq    %r12, %rsi

    movsd   0(%rdx), %xmm0
    movsd   8(%rdx), %xmm1
    movsd   16(%rdx), %xmm2
    movsd   24(%rdx), %xmm3

    movl    $4, %eax

    call    printf@PLT

.L_loop_inc:

    incq    %r12
    jmp     .L_print_loop_body

# =============================================================================
# 20. CLEANUP
# =============================================================================

.L_cleanup:

    # Free GPU output
    movq    d_out_ptr(%rip), %rdi
    testq   %rdi, %rdi
    jz      .L_cleanup_input

    call    cuMemFree_v2@PLT

.L_cleanup_input:

    movq    d_in_ptr(%rip), %rdi
    testq   %rdi, %rdi
    jz      .L_cleanup_mapping

    call    cuMemFree_v2@PLT

.L_cleanup_mapping:

    movq    mapped_ptr(%rip), %rdi
    cmpq    $0, %rdi
    je      .L_cleanup_fd

    movq    mapped_size(%rip), %rsi
    movq    $SYS_MUNMAP, %rax
    syscall

.L_cleanup_fd:

    movq    input_fd(%rip), %rdi
    cmpq    $0, %rdi
    jl      .L_cleanup_output_fd

    movq    $SYS_CLOSE, %rax
    syscall

.L_cleanup_output_fd:

    movq    out_fd(%rip), %rdi
    cmpq    $1, %rdi
    je      .L_success_exit

    movq    $SYS_CLOSE, %rax
    syscall

.L_success_exit:

    movq    %rbp, %rsp
    popq    %rbp

    xorq    %rdi, %rdi

    movq    $SYS_EXIT_GROUP, %rax
    syscall

# =============================================================================
# ERROR: USAGE
# =============================================================================

.L_usage:

    # We cannot safely call libc yet because the stack is at process-entry
    # alignment. Use a minimal syscall write instead.

    leaq    usage_msg(%rip), %rsi

    # For simplicity, fall through to framed version.

    pushq   %rbp
    movq    %rsp, %rbp
    andq    $-16, %rsp
    subq    $32, %rsp

    leaq    usage_msg(%rip), %rdi
    movq    %r11, %rsi
    xorl    %eax, %eax
    call    printf@PLT

    movl    $1, %edi
    jmp     .L_exit_error

# =============================================================================
# ERROR: USAGE AFTER FRAME
# =============================================================================

.L_usage_after_frame:

    leaq    usage_msg(%rip), %rdi
    movq    8(%rbp), %rsi
    xorl    %eax, %eax
    call    printf@PLT

    movl    $1, %edi
    jmp     .L_exit_error

# =============================================================================
# ERROR HANDLERS
# =============================================================================

.L_error_open_input:

    leaq    err_open_input(%rip), %rdi
    jmp     .L_print_error_exit

.L_error_open_output:

    leaq    err_open_input(%rip), %rdi
    jmp     .L_print_error_exit

.L_error_mmap:

    leaq    err_mmap(%rip), %rdi
    jmp     .L_print_error_exit

.L_error_malloc:

    leaq    err_malloc(%rip), %rdi
    jmp     .L_print_error_exit

.L_error_empty:

    leaq    err_empty(%rip), %rdi
    jmp     .L_print_error_exit

.L_error_too_many:

    leaq    err_too_many(%rip), %rdi
    movq    $MAX_ROWS, %rsi
    xorl    %eax, %eax
    call    printf@PLT
    movl    $1, %edi
    jmp     .L_exit_error

.L_parse_error:

    leaq    err_parse(%rip), %rdi
    movq    %rbx, %rsi
    xorl    %eax, %eax
    call    printf@PLT
    movl    $1, %edi
    jmp     .L_exit_error

# =============================================================================
# CUDA ERROR HANDLERS
# =============================================================================

.L_cuda_init_error:
    leaq    err_init(%rip), %rdi
    movl    %eax, %esi
    jmp     .L_cuda_report_exit

.L_cuda_device_error:
    leaq    err_device(%rip), %rdi
    movl    %eax, %esi
    jmp     .L_cuda_report_exit

.L_cuda_context_error:
    leaq    err_context(%rip), %rdi
    movl    %eax, %esi
    jmp     .L_cuda_report_exit

.L_cuda_module_error:
    leaq    err_module(%rip), %rdi
    movl    %eax, %esi
    jmp     .L_cuda_report_exit

.L_cuda_function_error:
    leaq    err_function(%rip), %rdi
    movl    %eax, %esi
    jmp     .L_cuda_report_exit

.L_cuda_alloc_error:
    leaq    err_alloc(%rip), %rdi
    movl    %eax, %esi
    jmp     .L_cuda_report_exit

.L_cuda_copy_error:
    leaq    err_copy(%rip), %rdi
    movl    %eax, %esi
    jmp     .L_cuda_report_exit

.L_cuda_launch_error:
    leaq    err_launch(%rip), %rdi
    movl    %eax, %esi
    jmp     .L_cuda_report_exit

.L_cuda_sync_error:
    leaq    err_sync(%rip), %rdi
    movl    %eax, %esi
    jmp     .L_cuda_report_exit

# -----------------------------------------------------------------------------
# Print CUDA error and exit.
#
# rdi = our message
# esi = CUresult
# -----------------------------------------------------------------------------

.L_cuda_report_exit:

    movl    %esi, cuda_error_code(%rip)

    # Print primary error
    movq    %rdi, %r12

    movq    %r12, %rdi
    xorl    %eax, %eax
    call    printf@PLT

    # Print numeric CUDA error
    leaq    err_cuda_code(%rip), %rdi
    movq    cuda_error_code(%rip), %rsi
    xorl    %eax, %eax
    call    printf@PLT

    # -------------------------------------------------------------------------
    # cuGetErrorName()
    #
    # CUresult cuGetErrorName(
    #     CUresult error,
    #     const char **pStr
    # )
    # -------------------------------------------------------------------------

    leaq    cuda_error_name_ptr(%rip), %rsi
    movl    cuda_error_code(%rip), %edi

    call    cuGetErrorName@PLT

    testl   %eax, %eax
    jnz     .L_cuda_string

    leaq    err_cuda_name(%rip), %rdi
    movq    cuda_error_name_ptr(%rip), %rsi
    xorl    %eax, %eax
    call    printf@PLT

.L_cuda_string:

    # -------------------------------------------------------------------------
    # cuGetErrorString()
    # -------------------------------------------------------------------------

    leaq    cuda_error_string_ptr(%rip), %rsi
    movl    cuda_error_code(%rip), %edi

    call    cuGetErrorString@PLT

    testl   %eax, %eax
    jnz     .L_cuda_exit

    leaq    err_cuda_string(%rip), %rdi
    movq    cuda_error_string_ptr(%rip), %rsi
    xorl    %eax, %eax
    call    printf@PLT

.L_cuda_exit:

    movl    $1, %edi
    jmp     .L_exit_error

# =============================================================================
# GENERIC ERROR
# =============================================================================

.L_print_error_exit:

    xorl    %eax, %eax
    call    printf@PLT

    movl    $1, %edi

.L_exit_error:

    movq    %rbp, %rsp
    popq    %rbp

.L_exit:

    movq    $SYS_EXIT_GROUP, %rax
    syscall

# =============================================================================
# LOCAL strcmp
# =============================================================================

.type strcmp_local,@function

strcmp_local:

    xorl    %eax, %eax

.L_strcmp_loop:

    movb    (%rdi), %dl
    movb    (%rsi), %cl

    cmpb    %cl, %dl
    jne     .L_strcmp_diff

    testb   %dl, %dl
    jz      .L_strcmp_done

    incq    %rdi
    incq    %rsi

    jmp     .L_strcmp_loop

.L_strcmp_diff:

    movl    $1, %eax
    ret

.L_strcmp_done:

    xorl    %eax, %eax
    ret

.size strcmp_local, . - strcmp_local

.section .note.GNU-stack,"",@progbits
