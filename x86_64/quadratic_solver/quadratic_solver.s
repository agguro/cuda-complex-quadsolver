# -----------------------------------------------------------------------------
# File: main.s (Double Precision Upgrade)
# Purpose: Headerless GPU Execution with ABI-compliant stack alignment (f64)
# -----------------------------------------------------------------------------

.equ CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 1
.equ CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 16

.equ DOUBLE_SIZE,       8   
.equ PADDED_COEFFS,     8   
.equ OUTPUT_ROOTS,      4   
.equ SIZEOF_INPUT_ROW,  (PADDED_COEFFS * DOUBLE_SIZE)  # 64 Bytes
.equ SIZEOF_OUTPUT_ROW, (PADDED_COEFFS * DOUBLE_SIZE)  
.equ MAX_ROWS,          1024
.equ HOST_IN_BUF_SIZE,  (MAX_ROWS * SIZEOF_INPUT_ROW)  
.equ HOST_OUT_BUF_SIZE, (MAX_ROWS * SIZEOF_OUTPUT_ROW) 

.equ SYS_OPEN,    2
.equ SYS_MMAP,    9
.equ SYS_MUNMAP,  11
.equ SYS_CLOSE,   3
.equ O_RDONLY,    0
.equ PROT_READ,   1
.equ MAP_PRIVATE, 2

.section .rodata
    .align 16
    kernel_bin:   .incbin "quadratic_solver.cubin"
    kernel_name:  .asciz  "quadratic_solver"
    usage_msg:    .asciz  "Usage: %s <input.csv> [-o <output.csv>]\n"
    opt_o:        .asciz  "-o"
    csv_format:   .asciz  "%lf,%lf,%lf,%lf,%lf,%lf\n"
    csv_row_fmt:  .asciz  "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n"
    res_fmt:      .asciz  "Row %ld: Roots -> R1: (%+.4f, %+.4fi) | R2: (%+.4f, %+.4fi)\n"
    hdr_msg:      .asciz  "\n--- GPU Execution Results (Double Precision) ---\n"

.section .data
    .align 16
    d_in_ptr:      .quad 0
    d_out_ptr:     .quad 0
    h_module:      .quad 0
    h_func:        .quad 0
    out_fd:        .quad 1                                     
    k_params:      .quad kparam_in_ptr, kparam_out_ptr, kparam_N
    kparam_in_ptr:  .quad 0
    kparam_out_ptr: .quad 0
    kparam_N:       .quad 0

.section .text
.globl _start
_start:
    # --- 1. Argument Parsing ---
    movq    (%rsp), %r15                       
    movq    8(%rsp), %r11                      
    cmpq    $2, %r15
    jl      .L_usage

    movq    $2, %r12
.L_arg_parse_loop:
    cmpq    %r15, %r12
    jge     .L_arg_parse_done
    movq    8(%rsp, %r12, 8), %rdi             
    leaq    opt_o(%rip), %rsi
    call    strcmp_local
    testl   %eax, %eax
    jnz     .L_next_arg
    incq    %r12
    cmpq    %r15, %r12
    jge     .L_usage
    movq    8(%rsp, %r12, 8), %rdi             
    movq    $577, %rsi                         
    movq    $438, %rdx                         
    movq    $SYS_OPEN, %rax
    syscall
    movq    %rax, out_fd(%rip)
.L_next_arg:
    incq    %r12
    jmp     .L_arg_parse_loop

.L_usage:
    leaq    usage_msg(%rip), %rdi
    movq    %r11, %rsi
    xorl    %eax, %eax
    call    printf@PLT
    movl    $1, %edi
    jmp     .L_exit

.L_arg_parse_done:
    movq    16(%rsp), %r12                     
    pushq   %rbp
    movq    %rsp, %rbp
    andq    $-16, %rsp                         
    subq    $256, %rsp                         

    # --- 2. CUDA Setup ---
    xorl    %edi, %edi
    call    cuInit@PLT
    leaq    128(%rsp), %rdi
    xorl    %esi, %esi
    call    cuDeviceGet@PLT
    leaq    136(%rsp), %rdi
    xorl    %esi, %esi
    movl    128(%rsp), %edx
    call    cuCtxCreate_v2@PLT
    leaq    h_module(%rip), %rdi
    leaq    kernel_bin(%rip), %rsi
    call    cuModuleLoadData@PLT
    leaq    h_func(%rip), %rdi
    movq    h_module(%rip), %rsi
    leaq    kernel_name(%rip), %rdx
    call    cuModuleGetFunction@PLT

    # --- 3. Memory & File Mapping ---
    movq    %r12, %rdi
    movq    $O_RDONLY, %rsi
    xorq    %rdx, %rdx
    movq    $SYS_OPEN, %rax
    syscall
    movq    %rax, %r12                         

    xorq    %rdi, %rdi
    movq    $65536, %rsi
    movq    $PROT_READ, %rdx
    movq    $MAP_PRIVATE, %r10
    movq    %r12, %r8
    xorq    %r9, %r9
    movq    $SYS_MMAP, %rax
    syscall
    movq    %rax, %r15                         

    movq    $HOST_IN_BUF_SIZE, %rdi
    call    malloc@PLT
    movq    %rax, %r13                         
    movq    $HOST_OUT_BUF_SIZE, %rdi
    call    malloc@PLT
    movq    %rax, %r14                         

    # --- 4. Parsing ---
    movq    %r15, %r12                         
    xorq    %rbx, %rbx                         

.Lscan_loop:
    cmpq    $MAX_ROWS, %rbx
    je      .Lscan_done
    movq    %rbx, %rax
    imulq   $SIZEOF_INPUT_ROW, %rax
    addq    %r13, %rax

    movq    %r12, %rdi
    leaq    csv_format(%rip), %rsi
    
    movq    %rax, %rdx         
    leaq    8(%rax), %rcx      
    leaq    16(%rax), %r8      
    leaq    24(%rax), %r9      

    subq    $16, %rsp
    leaq    32(%rax), %r10     
    movq    %r10, 0(%rsp)
    leaq    40(%rax), %r10     
    movq    %r10, 8(%rsp)
    xorl    %eax, %eax
    call    sscanf@PLT
    addq    $16, %rsp
    cmpq    $6, %rax
    jne     .Lscan_done

.Lnext_newline:
    cmpb    $10, (%r12)
    je      .Lnext_row
    cmpb    $0, (%r12)
    je      .Llast_row_done
    incq    %r12
    jmp     .Lnext_newline

.Lnext_row:
    incq    %r12
    incq    %rbx
    jmp     .Lscan_loop

.Llast_row_done:
    incq    %rbx   # Correct count for the final row

.Lscan_done:
    # --- 5. GPU Execution ---
    leaq    d_in_ptr(%rip), %rdi
    movq    %rbx, %rsi
    imulq   $SIZEOF_INPUT_ROW, %rsi
    call    cuMemAlloc_v2@PLT
    leaq    d_out_ptr(%rip), %rdi
    movq    %rbx, %rsi
    imulq   $SIZEOF_OUTPUT_ROW, %rsi
    call    cuMemAlloc_v2@PLT

    movq    d_in_ptr(%rip), %rdi
    movq    %r13, %rsi
    movq    %rbx, %rdx
    imulq   $SIZEOF_INPUT_ROW, %rdx
    call    cuMemcpyHtoD_v2@PLT

    movq    d_in_ptr(%rip), %rax
    movq    %rax, kparam_in_ptr(%rip)
    movq    d_out_ptr(%rip), %rax
    movq    %rax, kparam_out_ptr(%rip)
    movq    %rbx, kparam_N(%rip)

    movq    h_func(%rip), %rdi
    movl    $1, %esi                                
    movl    $1, %edx
    movl    $1, %ecx
    movl    %ebx, %r8d                              
    movl    $1, %r9d
    
    subq    $48, %rsp
    movq    $1, 0(%rsp)     
    movq    $1, 8(%rsp)     
    movq    $0, 16(%rsp)    
    leaq    k_params(%rip), %rax
    movq    %rax, 24(%rsp)  
    movq    $0, 32(%rsp)
    movq    $0, 40(%rsp)
    call    cuLaunchKernel@PLT
    addq    $48, %rsp
    call    cuCtxSynchronize@PLT

    movq    %r14, %rdi
    movq    d_out_ptr(%rip), %rsi
    movq    %rbx, %rdx
    imulq   $SIZEOF_OUTPUT_ROW, %rdx
    call    cuMemcpyDtoH_v2@PLT

    # --- 6. Printing Results ---
    movq    out_fd(%rip), %r15
    cmpq    $1, %r15
    jne     .L_print_prep
    leaq    hdr_msg(%rip), %rdi
    call    printf@PLT

.L_print_prep:
    xorq    %r12, %r12
.L_print_loop:
    cmpq    %rbx, %r12
    je      .L_cleanup
    
    movq    %r12, %rax
    imulq   $SIZEOF_INPUT_ROW, %rax
    addq    %r13, %rax
    movq    %rax, %rcx                              
    
    movq    %r12, %rax
    imulq   $SIZEOF_OUTPUT_ROW, %rax
    addq    %r14, %rax
    movq    %rax, %rdx                              

    cmpq    $1, %r15
    je      .L_print_terminal

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
    movl    $8, %eax
    call    dprintf@PLT
    addq    $16, %rsp
    jmp     .L_loop_inc

.L_print_terminal:
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
    jmp     .L_print_loop

.L_cleanup:
    movq    %rbp, %rsp
    popq    %rbp
    xorl    %edi, %edi
.L_exit:
    movq    $231, %rax
    syscall

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
    sbbl    %eax, %eax
    orl     $1, %eax
.L_sdone:
    ret

.size _start, . - _start
.section .note.GNU-stack,"",@progbits
