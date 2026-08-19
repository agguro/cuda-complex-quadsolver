# -----------------------------------------------------------------------------
# File: main.s
#
# x86-64 Linux CUDA Driver API host
#
# Pipeline:
#
#     test.csv
#        |
#        v
#     mmap()
#        |
#        v
#     sscanf()
#        |
#        v
#     host input buffer
#        |
#        | cuMemcpyHtoD
#        v
#     GPU input buffer
#        |
#        | cuLaunchKernel
#        v
#     GPU quadratic_solver
#        |
#        | cuMemcpyDtoH
#        v
#     host output buffer
#        |
#        v
#     printf / dprintf
#
# Error handling:
#   - Linux syscalls checked
#   - CUDA Driver API calls checked
#   - CUDA error name + description printed
#   - program terminates immediately on fatal errors
#
# -----------------------------------------------------------------------------

.equ DOUBLE_SIZE,       8
.equ PADDED_COEFFS,     8
.equ SIZEOF_INPUT_ROW,  (PADDED_COEFFS * DOUBLE_SIZE)     # 64 bytes
.equ SIZEOF_OUTPUT_ROW, (PADDED_COEFFS * DOUBLE_SIZE)     # 64 bytes

.equ MAX_ROWS,          1024

.equ HOST_IN_BUF_SIZE,  (MAX_ROWS * SIZEOF_INPUT_ROW)
.equ HOST_OUT_BUF_SIZE, (MAX_ROWS * SIZEOF_OUTPUT_ROW)

# Linux syscalls
.equ SYS_OPEN,          2
.equ SYS_CLOSE,         3
.equ SYS_FSTAT,         5
.equ SYS_MMAP,          9
.equ SYS_MUNMAP,        11
.equ SYS_EXIT_GROUP,    231

# open()
.equ O_RDONLY,          0
.equ O_WRONLY,          1
.equ O_CREAT,           64
.equ O_TRUNC,           512

# mmap()
.equ PROT_READ,         1
.equ MAP_PRIVATE,       2

# fstat structure
# x86-64 Linux:
# struct stat.st_size = offset 48
.equ STAT_SIZE_OFFSET,  48
.equ STAT_BUFFER_SIZE,  144

# -----------------------------------------------------------------------------
# READONLY DATA
# -----------------------------------------------------------------------------

.section .rodata
.align 16

# Embedded CUDA binary
kernel_bin:
    .incbin "quadratic_solver.cubin"

kernel_name:
    .asciz "quadratic_solver"

# Command line
usage_msg:
    .asciz "Usage: %s <input.csv> [-o <output.csv>]\n"

# General messages
msg_cuda_init:
    .asciz "cuInit"

msg_cuda_device:
    .asciz "cuDeviceGet"

msg_cuda_context:
    .asciz "cuCtxCreate"

msg_cuda_module:
    .asciz "cuModuleLoadData"

msg_cuda_function:
    .asciz "cuModuleGetFunction"

msg_cuda_alloc_input:
    .asciz "cuMemAlloc(input)"

msg_cuda_alloc_output:
    .asciz "cuMemAlloc(output)"

msg_cuda_htod:
    .asciz "cuMemcpyHtoD"

msg_cuda_launch:
    .asciz "cuLaunchKernel"

msg_cuda_sync:
    .asciz "cuCtxSynchronize"

msg_cuda_dtoh:
    .asciz "cuMemcpyDtoH"

msg_cuda_free_input:
    .asciz "cuMemFree(input)"

msg_cuda_free_output:
    .asciz "cuMemFree(output)"

msg_open:
    .asciz "open"

msg_fstat:
    .asciz "fstat"

msg_mmap:
    .asciz "mmap"

msg_malloc_input:
    .asciz "malloc(input)"

msg_malloc_output:
    .asciz "malloc(output)"

msg_bad_file:
    .asciz "Input file is empty or too small.\n"

msg_bad_size:
    .asciz "Input file size is not a multiple of a valid CSV record layout.\n"

msg_too_many:
    .asciz "Input contains more than MAX_ROWS rows.\n"

msg_bad_gpu_rows:
    .asciz "Row count exceeds the configured GPU block size.\n"

msg_parse:
    .asciz "CSV parsing stopped because a row could not be parsed.\n"

msg_cuda_error:
    .asciz "CUDA ERROR in %s\n"
msg_cuda_name:
    .asciz "  Name       : %s\n"
msg_cuda_desc:
    .asciz "  Description: %s\n"
msg_cuda_code:
    .asciz "  Code       : %d\n"

msg_sys_error:
    .asciz "SYSTEM ERROR in %s (errno=%ld)\n"

msg_header:
    .asciz "\n--- GPU Execution Results (Double Precision) ---\n"

res_fmt:
    .asciz "Row %ld: Roots -> R1: (%+.4f, %+.4fi) | R2: (%+.4f, %+.4fi)\n"

csv_row_fmt:
    .asciz "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n"

opt_o:
    .asciz "-o"

csv_format:
    .asciz "%lf,%lf,%lf,%lf,%lf,%lf\n"

.align 8
zero_double:
    .double 0.0


# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

.section .data
.align 16

# CUDA handles
d_in_ptr:
    .quad 0

d_out_ptr:
    .quad 0

h_module:
    .quad 0

h_func:
    .quad 0

# Input/output
input_fd:
    .quad -1

out_fd:
    .quad 1

mapped_ptr:
    .quad 0

mapped_size:
    .quad 0

# Host buffers
host_in_ptr:
    .quad 0

host_out_ptr:
    .quad 0

# Number of parsed rows
row_count:
    .quad 0

# CUDA device
device_id:
    .long 0

# CUDA context
context:
    .quad 0

# CUDA kernel parameters
k_params:
    .quad kparam_in_ptr
    .quad kparam_out_ptr
    .quad kparam_N

kparam_in_ptr:
    .quad 0

kparam_out_ptr:
    .quad 0

kparam_N:
    .quad 0

# stat structure
.align 16
file_stat:
    .skip STAT_BUFFER_SIZE

# Temporary CUDA error information
cuda_error_name:
    .quad 0

cuda_error_string:
    .quad 0


# -----------------------------------------------------------------------------
# TEXT
# -----------------------------------------------------------------------------

.section .text
.globl _start
.type _start,@function

_start:

    # -------------------------------------------------------------------------
    # 1. ESTABLISH ABI-CORRECT STACK
    # -------------------------------------------------------------------------

    pushq   %rbp
    movq    %rsp, %rbp

    andq    $-16, %rsp
    subq    $256, %rsp

    #
    # Preserve argc / argv.
    #
    # At process entry:
    #
    #   (%rbp + 8)   = argc
    #   (%rbp + 16)  = argv[0]
    #   (%rbp + 24)  = argv[1]
    #
    movq    8(%rbp), %r15
    movq    16(%rbp), %r11

    cmpq    $2, %r15
    jl      .L_usage


    # -------------------------------------------------------------------------
    # 2. PARSE COMMAND LINE
    #
    # ./quadratic_solver input.csv
    #
    # ./quadratic_solver input.csv -o output.csv
    # -------------------------------------------------------------------------

    #
    # argv[1] = input file
    #
    movq    24(%rbp), %r12

    #
    # Default output = stdout
    #
    movq    $1, out_fd(%rip)

    #
    # Scan optional arguments.
    #
    movq    $2, %r13

.L_arg_loop:

    cmpq    %r15, %r13
    jge     .L_args_done

    movq    8(%rbp,%r13,8), %rdi
    leaq    opt_o(%rip), %rsi
    call    strcmp_local

    testl   %eax, %eax
    jnz     .L_arg_next

    #
    # "-o" found.
    #
    incq    %r13

    cmpq    %r15, %r13
    jge     .L_usage

    movq    8(%rbp,%r13,8), %rdi

    #
    # open(output, O_WRONLY|O_CREAT|O_TRUNC, 0644)
    #
    movq    $O_WRONLY|O_CREAT|O_TRUNC, %rsi
    movq    $0644, %rdx
    movq    $SYS_OPEN, %rax
    syscall

    testq   %rax, %rax
    js      .L_error_open_output

    movq    %rax, out_fd(%rip)

.L_arg_next:

    incq    %r13
    jmp     .L_arg_loop


.L_args_done:


    # -------------------------------------------------------------------------
    # 3. OPEN INPUT FILE
    # -------------------------------------------------------------------------

    movq    %r12, %rdi
    movq    $O_RDONLY, %rsi
    xorq    %rdx, %rdx

    movq    $SYS_OPEN, %rax
    syscall

    testq   %rax, %rax
    js      .L_error_open_input

    movq    %rax, input_fd(%rip)


    # -------------------------------------------------------------------------
    # 4. fstat()
    #
    # Determine the actual file size.
    # -------------------------------------------------------------------------

    movq    input_fd(%rip), %rdi
    leaq    file_stat(%rip), %rsi

    movq    $SYS_FSTAT, %rax
    syscall

    testq   %rax, %rax
    js      .L_error_fstat

    movq    STAT_SIZE_OFFSET+file_stat(%rip), %rax
    movq    %rax, mapped_size(%rip)

    #
    # Reject empty file.
    #
    testq   %rax, %rax
    jz      .L_error_empty_file


    # -------------------------------------------------------------------------
    # 5. mmap()
    # -------------------------------------------------------------------------

    xorq    %rdi, %rdi

    movq    mapped_size(%rip), %rsi

    movq    $PROT_READ, %rdx
    movq    $MAP_PRIVATE, %r10

    movq    input_fd(%rip), %r8
    xorq    %r9, %r9

    movq    $SYS_MMAP, %rax
    syscall

    #
    # mmap returns negative errno on failure.
    #
    testq   %rax, %rax
    js      .L_error_mmap

    movq    %rax, mapped_ptr(%rip)


    # -------------------------------------------------------------------------
    # 6. ALLOCATE HOST INPUT BUFFER
    # -------------------------------------------------------------------------

    movq    $HOST_IN_BUF_SIZE, %rdi
    call    malloc@PLT

    testq   %rax, %rax
    jz      .L_error_malloc_input

    movq    %rax, host_in_ptr(%rip)


    # -------------------------------------------------------------------------
    # 7. ALLOCATE HOST OUTPUT BUFFER
    # -------------------------------------------------------------------------

    movq    $HOST_OUT_BUF_SIZE, %rdi
    call    malloc@PLT

    testq   %rax, %rax
    jz      .L_error_malloc_output

    movq    %rax, host_out_ptr(%rip)


    # -------------------------------------------------------------------------
    # 8. PARSE CSV
    #
    # Each input row:
    #
    #     a,b,c,0,0,0
    #
    # Stored as:
    #
    #     [a][b][c][padding][padding][padding][padding][padding]
    #
    # 8 doubles = 64 bytes.
    # -------------------------------------------------------------------------

    movq    mapped_ptr(%rip), %r12
    xorq    %r13, %r13


.L_scan_loop:

    cmpq    $MAX_ROWS, %r13
    jae     .L_error_too_many_rows

    #
    # destination = host_in_ptr + row * 64
    #
    movq    %r13, %rax
    imulq   $SIZEOF_INPUT_ROW, %rax
    addq    host_in_ptr(%rip), %rax

    #
    # sscanf(
    #   current,
    #   "%lf,%lf,%lf,%lf,%lf,%lf\n",
    #   &row[0],
    #   &row[1],
    #   ...
    #   &row[5]
    # )
    #
    movq    %r12, %rdi
    leaq    csv_format(%rip), %rsi

    movq    %rax, %rdx
    leaq    8(%rax), %rcx
    leaq    16(%rax), %r8
    leaq    24(%rax), %r9

    #
    # Arguments 7 and 8 go on stack.
    #
    subq    $16, %rsp

    leaq    32(%rax), %r10
    movq    %r10, 0(%rsp)

    leaq    40(%rax), %r10
    movq    %r10, 8(%rsp)

    xorl    %eax, %eax

    call    sscanf@PLT

    addq    $16, %rsp

    #
    # Six successfully converted values required.
    #
    cmpq    $6, %rax
    jne     .L_scan_finished


    # -------------------------------------------------------------------------
    # Find next line.
    # -------------------------------------------------------------------------

.L_find_newline:

    cmpb    $10, (%r12)
    je      .L_next_row

    cmpb    $0, (%r12)
    je      .L_last_row

    incq    %r12
    jmp     .L_find_newline


.L_next_row:

    incq    %r12
    incq    %r13
    jmp     .L_scan_loop


.L_last_row:

    #
    # Last row did not necessarily end in '\n'.
    #
    incq    %r13


.L_scan_finished:

    testq   %r13, %r13
    jz      .L_error_parse

    movq    %r13, row_count(%rip)


    # -------------------------------------------------------------------------
    # 9. CUDA INITIALIZATION
    # -------------------------------------------------------------------------

    xorl    %edi, %edi

    call    cuInit@PLT

    testl   %eax, %eax
    jnz     .L_cuda_init_error


    # -------------------------------------------------------------------------
    # 10. GET DEVICE 0
    # -------------------------------------------------------------------------

    leaq    device_id(%rip), %rdi
    xorl    %esi, %esi

    call    cuDeviceGet@PLT

    testl   %eax, %eax
    jnz     .L_cuda_device_error


    # -------------------------------------------------------------------------
    # 11. CREATE CUDA CONTEXT
    # -------------------------------------------------------------------------

    leaq    context(%rip), %rdi

    xorl    %esi, %esi

    movl    device_id(%rip), %edx

    call    cuCtxCreate_v2@PLT

    testl   %eax, %eax
    jnz     .L_cuda_context_error


    # -------------------------------------------------------------------------
    # 12. LOAD EMBEDDED CUBIN
    # -------------------------------------------------------------------------

    leaq    h_module(%rip), %rdi
    leaq    kernel_bin(%rip), %rsi

    call    cuModuleLoadData@PLT

    testl   %eax, %eax
    jnz     .L_cuda_module_error


    # -------------------------------------------------------------------------
    # 13. GET KERNEL FUNCTION
    # -------------------------------------------------------------------------

    leaq    h_func(%rip), %rdi
    movq    h_module(%rip), %rsi
    leaq    kernel_name(%rip), %rdx

    call    cuModuleGetFunction@PLT

    testl   %eax, %eax
    jnz     .L_cuda_function_error


    # -------------------------------------------------------------------------
    # 14. GPU INPUT ALLOCATION
    # -------------------------------------------------------------------------

    leaq    d_in_ptr(%rip), %rdi

    movq    row_count(%rip), %rsi
    imulq   $SIZEOF_INPUT_ROW, %rsi

    call    cuMemAlloc_v2@PLT

    testl   %eax, %eax
    jnz     .L_cuda_alloc_input_error


    # -------------------------------------------------------------------------
    # 15. GPU OUTPUT ALLOCATION
    # -------------------------------------------------------------------------

    leaq    d_out_ptr(%rip), %rdi

    movq    row_count(%rip), %rsi
    imulq   $SIZEOF_OUTPUT_ROW, %rsi

    call    cuMemAlloc_v2@PLT

    testl   %eax, %eax
    jnz     .L_cuda_alloc_output_error


    # -------------------------------------------------------------------------
    # 16. HOST -> DEVICE
    # -------------------------------------------------------------------------

    movq    d_in_ptr(%rip), %rdi
    movq    host_in_ptr(%rip), %rsi

    movq    row_count(%rip), %rdx
    imulq   $SIZEOF_INPUT_ROW, %rdx

    call    cuMemcpyHtoD_v2@PLT

    testl   %eax, %eax
    jnz     .L_cuda_htod_error


    # -------------------------------------------------------------------------
    # 17. PREPARE KERNEL PARAMETERS
    #
    # CUDA Driver API expects:
    #
    #     void *kernelParams[]
    #
    # where each entry points to the actual parameter.
    #
    # k_params:
    #
    #     [0] -> kparam_in_ptr
    #     [1] -> kparam_out_ptr
    #     [2] -> kparam_N
    # -------------------------------------------------------------------------

    movq    d_in_ptr(%rip), %rax
    movq    %rax, kparam_in_ptr(%rip)

    movq    d_out_ptr(%rip), %rax
    movq    %rax, kparam_out_ptr(%rip)

    movq    row_count(%rip), %rax
    movq    %rax, kparam_N(%rip)


    # -------------------------------------------------------------------------
    # 18. LAUNCH GPU
    #
    # grid  = (1,1,1)
    # block = (N,1,1)
    #
    # This means one GPU thread per CSV row.
    # -------------------------------------------------------------------------

    movq    h_func(%rip), %rdi

    movl    $1, %esi                    # gridDimX
    movl    $1, %edx                    # gridDimY
    movl    $1, %ecx                    # gridDimZ

    movl    row_count(%rip), %r8d       # blockDimX
    movl    $1, %r9d                    # blockDimY

    #
    # cuLaunchKernel has additional parameters:
    #
    #   blockDimZ
    #   sharedMemBytes
    #   stream
    #   kernelParams
    #   extra
    #
    # They are stack arguments after the first five register arguments.
    #

    subq    $48, %rsp

    movq    $1, 0(%rsp)                 # blockDimZ
    movq    $0, 8(%rsp)                 # sharedMemBytes
    movq    $0, 16(%rsp)                # stream

    leaq    k_params(%rip), %rax
    movq    %rax, 24(%rsp)              # kernelParams

    movq    $0, 32(%rsp)                # extra
    movq    $0, 40(%rsp)                # padding

    call    cuLaunchKernel@PLT

    addq    $48, %rsp

    testl   %eax, %eax
    jnz     .L_cuda_launch_error


    # -------------------------------------------------------------------------
    # 19. WAIT FOR GPU
    # -------------------------------------------------------------------------

    call    cuCtxSynchronize@PLT

    testl   %eax, %eax
    jnz     .L_cuda_sync_error


    # -------------------------------------------------------------------------
    # 20. DEVICE -> HOST
    # -------------------------------------------------------------------------

    movq    host_out_ptr(%rip), %rdi
    movq    d_out_ptr(%rip), %rsi

    movq    row_count(%rip), %rdx
    imulq   $SIZEOF_OUTPUT_ROW, %rdx

    call    cuMemcpyDtoH_v2@PLT

    testl   %eax, %eax
    jnz     .L_cuda_dtoh_error


    # -------------------------------------------------------------------------
    # 21. PRINT RESULTS
    # -------------------------------------------------------------------------

    movq    out_fd(%rip), %r15

    cmpq    $1, %r15
    jne     .L_print_rows

    leaq    msg_header(%rip), %rdi
    xorl    %eax, %eax
    call    printf@PLT


.L_print_rows:

    xorq    %r12, %r12


.L_print_loop:

    cmpq    row_count(%rip), %r12
    jae     .L_success


    #
    # input row address
    #
    movq    %r12, %rax
    imulq   $SIZEOF_INPUT_ROW, %rax
    addq    host_in_ptr(%rip), %rax

    movq    %rax, %rcx


    #
    # output row address
    #
    movq    %r12, %rax
    imulq   $SIZEOF_OUTPUT_ROW, %rax
    addq    host_out_ptr(%rip), %rax

    movq    %rax, %rdx


    # -------------------------------------------------------------------------
    # Terminal output
    # -------------------------------------------------------------------------

    cmpq    $1, %r15
    je      .L_print_terminal


    # -------------------------------------------------------------------------
    # CSV output
    #
    # Input:
    #   6 doubles
    #
    # Output:
    #   4 doubles
    #
    # Total:
    #   10 floating point values.
    # -------------------------------------------------------------------------

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

    #
    # SysV AMD64 varargs:
    # AL = number of vector registers used.
    #
    subq    $16, %rsp

    movsd   %xmm8, 0(%rsp)
    movsd   %xmm9, 8(%rsp)

    movq    %r15, %rdi
    leaq    csv_row_fmt(%rip), %rsi

    movl    $8, %eax

    call    dprintf@PLT

    addq    $16, %rsp

    jmp     .L_loop_increment


.L_print_terminal:

    leaq    res_fmt(%rip), %rdi

    movq    %r12, %rsi

    movsd   0(%rdx), %xmm0
    movsd   8(%rdx), %xmm1
    movsd   16(%rdx), %xmm2
    movsd   24(%rdx), %xmm3

    movl    $4, %eax

    call    printf@PLT


.L_loop_increment:

    incq    %r12
    jmp     .L_print_loop


    # -------------------------------------------------------------------------
    # SUCCESS
    # -------------------------------------------------------------------------

.L_success:

    #
    # Free device input.
    #
    movq    d_in_ptr(%rip), %rdi
    testq   %rdi, %rdi
    jz      .L_free_output

    call    cuMemFree_v2@PLT

    #
    # We intentionally don't abort success path if cleanup fails.
    #


.L_free_output:

    movq    d_out_ptr(%rip), %rdi
    testq   %rdi, %rdi
    jz      .L_cleanup_host

    call    cuMemFree_v2@PLT


.L_cleanup_host:

    #
    # munmap()
    #
    movq    mapped_ptr(%rip), %rdi
    movq    mapped_size(%rip), %rsi

    testq   %rdi, %rdi
    jz      .L_close_input

    movq    $SYS_MUNMAP, %rax
    syscall


.L_close_input:

    movq    input_fd(%rip), %rdi
    cmpq    $0, %rdi
    jl      .L_free_host

    movq    $SYS_CLOSE, %rax
    syscall


.L_free_host:

    movq    host_in_ptr(%rip), %rdi
    testq   %rdi, %rdi
    jz      .L_free_output_host

    call    free@PLT


.L_free_output_host:

    movq    host_out_ptr(%rip), %rdi
    testq   %rdi, %rdi
    jz      .L_exit_success

    call    free@PLT


.L_exit_success:

    movq    %rbp, %rsp
    popq    %rbp

    xorl    %edi, %edi

    movq    $SYS_EXIT_GROUP, %rax
    syscall


# =============================================================================
# ERROR HANDLERS
# =============================================================================

.L_usage:

    leaq    usage_msg(%rip), %rdi
    movq    %r11, %rsi

    xorl    %eax, %eax
    call    printf@PLT

    movl    $1, %edi
    jmp     .L_fatal_exit


.L_error_open_input:

    leaq    msg_open(%rip), %rdi
    jmp     .L_sys_error


.L_error_open_output:

    leaq    msg_open(%rip), %rdi
    jmp     .L_sys_error


.L_error_fstat:

    leaq    msg_fstat(%rip), %rdi
    jmp     .L_sys_error


.L_error_mmap:

    leaq    msg_mmap(%rip), %rdi
    jmp     .L_sys_error


.L_error_malloc_input:

    leaq    msg_malloc_input(%rip), %rdi
    jmp     .L_simple_error


.L_error_malloc_output:

    leaq    msg_malloc_output(%rip), %rdi
    jmp     .L_simple_error


.L_error_empty_file:

    leaq    msg_bad_file(%rip), %rdi
    jmp     .L_simple_error


.L_error_too_many_rows:

    leaq    msg_too_many(%rip), %rdi
    jmp     .L_simple_error


.L_error_parse:

    leaq    msg_parse(%rip), %rdi
    jmp     .L_simple_error


# -----------------------------------------------------------------------------
# CUDA ERROR ROUTING
#
# At each entry:
#
#     eax = CUresult
#
# We put it into r12 and call cuda_error().
# -----------------------------------------------------------------------------

.L_cuda_init_error:

    movl    %eax, %r12d
    leaq    msg_cuda_init(%rip), %rdi
    jmp     cuda_error


.L_cuda_device_error:

    movl    %eax, %r12d
    leaq    msg_cuda_device(%rip), %rdi
    jmp     cuda_error


.L_cuda_context_error:

    movl    %eax, %r12d
    leaq    msg_cuda_context(%rip), %rdi
    jmp     cuda_error


.L_cuda_module_error:

    movl    %eax, %r12d
    leaq    msg_cuda_module(%rip), %rdi
    jmp     cuda_error


.L_cuda_function_error:

    movl    %eax, %r12d
    leaq    msg_cuda_function(%rip), %rdi
    jmp     cuda_error


.L_cuda_alloc_input_error:

    movl    %eax, %r12d
    leaq    msg_cuda_alloc_input(%rip), %rdi
    jmp     cuda_error


.L_cuda_alloc_output_error:

    movl    %eax, %r12d
    leaq    msg_cuda_alloc_output(%rip), %rdi
    jmp     cuda_error


.L_cuda_htod_error:

    movl    %eax, %r12d
    leaq    msg_cuda_htod(%rip), %rdi
    jmp     cuda_error


.L_cuda_launch_error:

    movl    %eax, %r12d
    leaq    msg_cuda_launch(%rip), %rdi
    jmp     cuda_error


.L_cuda_sync_error:

    movl    %eax, %r12d
    leaq    msg_cuda_sync(%rip), %rdi
    jmp     cuda_error


.L_cuda_dtoh_error:

    movl    %eax, %r12d
    leaq    msg_cuda_dtoh(%rip), %rdi
    jmp     cuda_error


# -----------------------------------------------------------------------------
# CUDA ERROR REPORTER
#
# Input:
#   r12d = CUresult
#   rdi  = operation name
# -----------------------------------------------------------------------------

cuda_error:

    #
    # Save operation name.
    #
    movq    %rdi, %r13

    #
    # Save CUresult.
    #
    movl    %r12d, %r14d

    #
    # cuGetErrorName(result, &name)
    #
    movl    %r14d, %edi
    leaq    cuda_error_name(%rip), %rsi

    call    cuGetErrorName@PLT

    #
    # cuGetErrorString(result, &string)
    #
    movl    %r14d, %edi
    leaq    cuda_error_string(%rip), %rsi

    call    cuGetErrorString@PLT

    #
    # Header
    #
    leaq    msg_cuda_error(%rip), %rdi
    movq    %r13, %rsi

    xorl    %eax, %eax
    call    printf@PLT

    #
    # Name
    #
    leaq    msg_cuda_name(%rip), %rdi
    movq    cuda_error_name(%rip), %rsi

    xorl    %eax, %eax
    call    printf@PLT

    #
    # Description
    #
    leaq    msg_cuda_desc(%rip), %rdi
    movq    cuda_error_string(%rip), %rsi

    xorl    %eax, %eax
    call    printf@PLT

    #
    # Numeric code
    #
    leaq    msg_cuda_code(%rip), %rdi
    movl    %r14d, %esi

    xorl    %eax, %eax
    call    printf@PLT

    movl    $1, %edi
    jmp     .L_fatal_exit


# -----------------------------------------------------------------------------
# Generic syscall error
#
# rdi = operation string
# syscall result is assumed to be in rax.
# -----------------------------------------------------------------------------

.L_sys_error:

    movq    %rdi, %r13

    #
    # Linux syscall failure:
    #   rax = -errno
    #
    negq    %rax
    movq    %rax, %r14

    leaq    msg_sys_error(%rip), %rdi
    movq    %r13, %rsi
    movq    %r14, %rdx

    xorl    %eax, %eax
    call    printf@PLT

    movl    $1, %edi
    jmp     .L_fatal_exit


# -----------------------------------------------------------------------------
# Simple error message
#
# rdi = message
# -----------------------------------------------------------------------------

.L_simple_error:

    xorl    %eax, %eax
    call    printf@PLT

    movl    $1, %edi
    jmp     .L_fatal_exit


# -----------------------------------------------------------------------------
# FINAL EXIT
# -----------------------------------------------------------------------------

.L_fatal_exit:

    movq    %rbp, %rsp
    popq    %rbp

    movq    $SYS_EXIT_GROUP, %rax
    syscall


# =============================================================================
# LOCAL strcmp()
#
# int strcmp_local(const char *a, const char *b)
#
# Returns:
#
#   0 = equal
#   1 = different
# =============================================================================

.type strcmp_local,@function

strcmp_local:

    xorl    %eax, %eax

.L_sloop:

    movb    (%rdi), %dl
    movb    (%rsi), %cl

    cmpb    %cl, %dl
    jne     .L_sdiff

    testb   %dl, %dl
    jz      .L_sdone

    incq    %rdi
    incq    %rsi

    jmp     .L_sloop


.L_sdiff:

    movl    $1, %eax


.L_sdone:

    ret


.size strcmp_local, .-strcmp_local
.size _start, .-_start

.section .note.GNU-stack,"",@progbits
