# ==============================================================================
# Project:     CUDA Complex Quadratic Solver Host Orchestrator
# File:        complex_quad_solver_host.s
# Author:      agguro
# Date:        August 20, 2026 
# Description: x86_64 host orchestrator for complex quadratic equation solver.
#              Safe RBP-based Argument Parsing & Full CUDA Driver Initialization.
# Architecture: x86_64 | Linux SysV ABI | AT&T Syntax
#
# Copyright 2026 agguro
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

.equ CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 1
.equ CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 16

.equ DOUBLE_SIZE,       8   
.equ PADDED_COEFFS,     8   
.equ OUTPUT_ROOTS,      4   
.equ SIZEOF_INPUT_ROW,  (PADDED_COEFFS * DOUBLE_SIZE)  # 64 Bytes
.equ SIZEOF_OUTPUT_ROW, (PADDED_COEFFS * DOUBLE_SIZE)  
.equ MAX_ROWS,          16384   # Scaled to handle 10k+ rows
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
    kernel_bin:    .incbin "complex_quadratic_solver.cubin"
    kernel_name:   .asciz  "complex_quadratic_solver"
    usage_msg:     .asciz  "Usage: %s <input.csv> [-o <output.csv>]\n"
    opt_o:         .asciz  "-o"
    csv_format:    .asciz  "%lf,%lf,%lf,%lf,%lf,%lf"
    csv_row_fmt:   .asciz  "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n"
    res_fmt:       .asciz  "Row %ld: Roots -> R1: (%+.4f, %+.4fi) | R2: (%+.4f, %+.4fi)\n"
    hdr_msg:       .asciz  "\n--- GPU Execution Results (Double Precision) ---\n"

.section .data
    .align 16
    d_in_ptr:      .quad 0
    d_out_ptr:     .quad 0
    h_module:      .quad 0
    h_func:        .quad 0
    out_fd:        .quad 1                                     
    k_params:      .quad kparam_in_ptr, kparam_out_ptr, kparam_N
    kparam_in_ptr: .quad 0
    kparam_out_ptr:.quad 0
    kparam_N:      .quad 0

.section .text
.globl _start
_start:
    # --- 1. Argument Parsing ---
    movq    (%rsp), %r15                    # Load argc into %r15
    movq    8(%rsp), %r11                   # Load argv[0] (program name) into %r11
    cmpq    $2, %r15                        # Compare argc with 2
    jl      .L_usage                        # Jump to usage if argc < 2

    movq    $2, %r12                        # Initialize argument index loop counter to 2
.L_arg_parse_loop:
    cmpq    %r15, %r12                      # Compare loop counter with argc
    jge     .L_arg_parse_done               # Exit loop if counter >= argc
    movq    8(%rsp, %r12, 8), %rdi          # Load current argv[i] pointer into %rdi
    leaq    opt_o(%rip), %rsi               # Load pointer to "-o" option string into %rsi
    call    strcmp_local                    # Compare current argument with "-o"
    testl   %eax, %eax                      # Test if strings are equal (%eax == 0)
    jnz     .L_next_arg                     # Jump to next argument if not equal
    incq    %r12                            # Increment loop counter to get filename argument
    cmpq    %r15, %r12                      # Check if filename argument exists
    jge     .L_usage                        # Jump to usage if missing
    movq    8(%rsp, %r12, 8), %rdi          # Load filename pointer into %rdi
    movq    $577, %rsi                      # Set flags for open (O_CREAT | O_WRONLY | O_TRUNC)
    movq    $438, %rdx                      # Set file permissions to 0666
    movq    $SYS_OPEN, %rax                 # Set syscall number for open
    syscall                                 # Execute sys_open syscall
    movq    %rax, out_fd(%rip)              # Store returned file descriptor into out_fd variable
.L_next_arg:
    incq    %r12                            # Increment argument index counter
    jmp     .L_arg_parse_loop               # Repeat argument parsing loop

.L_usage:
    leaq    usage_msg(%rip), %rdi           # Load format string pointer for usage message
    movq    %r11, %rsi                      # Move program name pointer into %rsi
    xorl    %eax, %eax                      # Clear vector registers count for variadic call
    call    printf@PLT                      # Call printf to print usage instructions
    movl    $1, %edi                        # Set exit status code to 1
    jmp     .L_exit                         # Jump to program exit handler

.L_arg_parse_done:
    movq    16(%rsp), %r12                  # Load input filename argument (argv[1]) into %r12
    pushq   %rbp                            # Push previous base pointer onto stack
    movq    %rsp, %rbp                      # Establish new stack frame base pointer
    andq    $-16, %rsp                      # Align stack pointer to 16-byte boundary
    subq    $256, %rsp                      # Allocate 256 bytes of local scratch space on stack

    # --- 2. CUDA Setup ---
    xorl    %edi, %edi                      # Clear %edi for default flags parameter
    call    cuInit@PLT                      # Initialize CUDA driver API subsystem
    leaq    128(%rsp), %rdi                 # Load stack address for device handle destination
    xorl    %esi, %esi                      # Select device ordinal 0
    call    cuDeviceGet@PLT                 # Retrieve CUDA device handle by ordinal
    leaq    136(%rsp), %rdi                 # Load stack address for context handle destination
    xorl    %esi, %esi                      # Clear flags parameter for context creation
    movl    128(%rsp), %edx                 # Load device handle into %edx parameter register
    call    cuCtxCreate_v2@PLT              # Create and activate CUDA primary context
    leaq    h_module(%rip), %rdi            # Load address of h_module pointer variable
    leaq    kernel_bin(%rip), %rsi          # Load address of embedded cubin binary payload
    call    cuModuleLoadData@PLT            # Load CUDA module image into memory
    leaq    h_func(%rip), %rdi              # Load address of h_func pointer variable
    movq    h_module(%rip), %rsi            # Load loaded module handle into %rsi
    leaq    kernel_name(%rip), %rdx         # Load kernel entry point symbol name string
    call    cuModuleGetFunction@PLT         # Retrieve function handle for kernel execution

    # --- 3. Memory & File Mapping & Dynamic Sizing ---
    movq    %r12, %rdi                      # Move input filename string pointer into %rdi
    movq    $O_RDONLY, %rsi                 # Set read-only access mode flag
    xorq    %rdx, %rdx                      # Clear mode parameter (not used for opening existing file)
    movq    $SYS_OPEN, %rax                 # Set syscall number for open
    syscall                                 # Execute open system call
    movq    %rax, %r12                      # Store returned file descriptor into %r12

    subq    $144, %rsp                      # Allocate stack space for stat buffer structure
    movq    %r12, %rdi                      # Move file descriptor into %rdi
    movq    %rsp, %rsi                      # Move pointer to stack stat buffer into %rsi
    movq    $5, %rax                        # Set syscall number for fstat (5)
    syscall                                 # Execute fstat system call to query file metadata

    movq    48(%rsp), %r8                   # Extract file size field from stat buffer into %r8
    addq    $144, %rsp                      # Restore stack pointer after stat allocation

    xorq    %rdi, %rdi                      # Clear address hint for mmap (kernel chooses address)
    movq    %r8, %rsi                       # Set mapping length to exact file size in bytes
    movq    $PROT_READ, %rdx                # Set protection flags to read-only access
    movq    $MAP_PRIVATE, %r10              # Set mapping flags to private copy-on-write
    movq    %r12, %r8                       # Pass valid input file descriptor into %r8
    xorq    %r9, %r9                        # Set file mapping offset to 0 bytes
    movq    $SYS_MMAP, %rax                 # Set syscall number for mmap
    syscall                                 # Execute mmap system call to map input file
    movq    %rax, %r15                      # Store mapped memory buffer base address in %r15

    movq    $HOST_IN_BUF_SIZE, %rdi         # Load requested input buffer byte size into %rdi
    call    malloc@PLT                      # Allocate host input buffer via standard malloc
    movq    %rax, %r13                      # Save host input buffer pointer in %r13

    movq    $HOST_OUT_BUF_SIZE, %rdi        # Load requested output buffer byte size into %rdi
    call    malloc@PLT                      # Allocate host output buffer via standard malloc
    movq    %rax, %r14                      # Save host output buffer pointer in %r14
    
    # --- 4. Parsing ---
    movq    %r15, %r12                      # Initialize parsing cursor pointer with mmapped base address
    xorq    %rbx, %rbx                      # Clear row counter register %rbx to 0

.Lscan_loop:
    cmpq    $MAX_ROWS, %rbx                 # Compare current row index with MAX_ROWS limit
    jge     .Lscan_done                     # Jump out of parsing loop if maximum rows reached
    movq    %rbx, %rax                      # Copy row index into working register
    imulq   $SIZEOF_INPUT_ROW, %rax         # Multiply row index by row stride size in bytes
    addq    %r13, %rax                      # Compute exact target memory destination address inside buffer

    movq    %r12, %rdi                      # Load current file text parsing cursor into %rdi
    leaq    csv_format(%rip), %rsi          # Load parsing format string pointer into %rsi
    
    movq    %rax, %rdx                      # Set first double destination parameter pointer
    leaq    8(%rax), %rcx                   # Set second double destination parameter pointer
    leaq    16(%rax), %r8                   # Set third double destination parameter pointer
    leaq    24(%rax), %r9                   # Set fourth double destination parameter pointer

    subq    $16, %rsp                       # Allocate stack space for remaining variadic sscanf pointers
    leaq    32(%rax), %r10                  # Compute fifth double destination memory address
    movq    %r10, 0(%rsp)                   # Store fifth pointer onto stack frame
    leaq    40(%rax), %r10                  # Compute sixth double destination memory address
    movq    %r10, 8(%rsp)                   # Store sixth pointer onto stack frame
    xorl    %eax, %eax                      # Clear vector register count for sscanf variadic call
    call    sscanf@PLT                      # Parse CSV row values into target memory slots
    addq    $16, %rsp                       # Clean up temporary stack space
    
    cmpq    $6, %rax                        # Check if sscanf successfully parsed exactly 6 coefficients
    jne     .L_skip_invalid_line            # Skip line processing if parsing failed or row was malformed
    incq    %rbx                            # Increment successfully parsed valid row count

.L_skip_invalid_line:
    movb    (%r12), %al                     # Load current byte from input text stream into %al
    testb   %al, %al                        # Test if character byte is null terminator
    jz      .Lscan_done                     # Jump to termination if end of file reached
    incq    %r12                            # Advance text parsing cursor pointer by one byte
    cmpb    $10, %al                        # Compare character with ASCII line feed (LF = 10)
    je      .Lscan_loop                     # Return to parsing loop if newline encountered
    jmp     .L_skip_invalid_line            # Continue scanning forward until newline found
    
.L_find_newline:
    movb    (%r12), %al                     # Read current character byte from text pointer
    testb   %al, %al                        # Check for null terminator end of string condition
    jz      .Lscan_done                     # Exit scan loop if end of file encountered
    incq    %r12                            # Advance parsing pointer to next byte position
    cmpb    $10, %al                        # Check if byte matches ASCII LF character
    je      .Lscan_loop                     # Restart scan loop if line break found
    jmp     .L_find_newline                 # Loop back to continue seeking newline marker

.Lnext_newline:
    movb    (%r12), %al                     # Fetch byte from current stream pointer position
    testb   %al, %al                        # Verify if end of file marker reached
    jz      .Lscan_done                     # Terminate scanning loop if null byte met
    incq    %r12                            # Increment stream scanning pointer forward
    cmpb    $10, %al                        # Test for ASCII newline code value
    je      .Lnext_row                      # Proceed to next row handler if match found
    jmp     .Lnext_newline                  # Continue searching for end of line boundary

.Lnext_row:
    incq    %r12                            # Increment string scanning cursor position past newline
    incq    %rbx                            # Increment processed row counter value
    jmp     .Lscan_loop                     # Loop back to process subsequent input data row

.Llast_row_done:
    incq    %rbx                            # Correct count for the final row entry

.Lscan_done:
    # --- 5. GPU Execution ---
    leaq    d_in_ptr(%rip), %rdi            # Load address of device input pointer destination variable
    movq    %rbx, %rsi                      # Move total active row count into %rsi
    imulq   $SIZEOF_INPUT_ROW, %rsi         # Compute total required input allocation byte size
    call    cuMemAlloc_v2@PLT               # Allocate global GPU memory space for input coefficients
    leaq    d_out_ptr(%rip), %rdi           # Load address of device output pointer variable
    movq    %rbx, %rsi                      # Move total row count into %rsi
    imulq   $SIZEOF_OUTPUT_ROW, %rsi        # Compute total required output buffer allocation size
    call    cuMemAlloc_v2@PLT               # Allocate global GPU memory space for computed roots

    movq    d_in_ptr(%rip), %rdi            # Load destination device input buffer address into %rdi
    movq    %r13, %rsi                      # Load source host input memory buffer pointer into %rsi
    movq    %rbx, %rdx                      # Move active row count parameter into %rdx
    imulq   $SIZEOF_INPUT_ROW, %rdx         # Calculate total byte transfer block length
    call    cuMemcpyHtoD_v2@PLT             # Transfer input coefficient data from host memory to device

    movq    d_in_ptr(%rip), %rax            # Load allocated device input pointer value into %rax
    movq    %rax, kparam_in_ptr(%rip)       # Store input pointer into kernel parameter structure block
    movq    d_out_ptr(%rip), %rax           # Load allocated device output pointer value into %rax
    movq    %rax, kparam_out_ptr(%rip)      # Store output pointer into kernel parameter structure block
    movq    %rbx, kparam_N(%rip)            # Store total row count value into kernel parameter block

    movq    h_func(%rip), %rdi              # Load compiled CUDA kernel function handle into %rdi
    movl    $1, %esi                        # Set grid dimension X size parameter to 1 block
    movl    $1, %edx                        # Set grid dimension Y size parameter to 1
    movl    $1, %ecx                        # Set grid dimension Z size parameter to 1
    movl    %ebx, %r8d                      # Set block dimension X thread count to match row count %ebx
    movl    $1, %r9d                        # Set block dimension Y thread count to 1
    
    subq    $48, %rsp                       # Allocate stack space for kernel launch extra configuration options
    movq    $1, 0(%rsp)                     # Set CU_LAUNCH_PARAM_BUFFER_POINTER option identifier tag
    movq    $1, 8(%rsp)                     # Set parameter buffer size specification indicator
    movq    $0, 16(%rsp)                    # Set extra parameter flag options to 0
    leaq    k_params(%rip), %rax            # Load address of kernel parameters argument array
    movq    %rax, 24(%rsp)                  # Store parameter array pointer into launch option block
    movq    $0, 32(%rsp)                    # Set extra option null terminator slot
    movq    $0, 40(%rsp)                    # Set padding/reserved option slot to zero
    call    cuLaunchKernel@PLT              # Dispatch and launch CUDA kernel execution asynchronously
    addq    $48, %rsp                       # Clean up temporary stack layout space after launch
    call    cuCtxSynchronize@PLT            # Synchronize host execution thread with CUDA device completion

    movq    %r14, %rdi                      # Load host output memory buffer pointer into %rdi
    movq    d_out_ptr(%rip), %rsi           # Load device output buffer base pointer into %rsi
    movq    %rbx, %rdx                      # Load total row count parameter into %rdx
    imulq   $SIZEOF_OUTPUT_ROW, %rdx        # Compute total transfer size in bytes
    call    cuMemcpyDtoH_v2@PLT             # Transfer computed result roots from device back to host memory

    # --- 6. Printing Results ---
    movq    out_fd(%rip), %r15              # Load output target file descriptor value into %r15
    cmpq    $1, %r15                        # Check if output descriptor points to standard output (stdout = 1)
    jne     .L_print_prep                   # Skip header banner printing if redirecting to file stream
    leaq    hdr_msg(%rip), %rdi             # Load address of results header message string
    call    printf@PLT                      # Print header block separator text to standard output

.L_print_prep:
    xorq    %r12, %r12                      # Reset print loop index counter register %r12 to zero
.L_print_loop:
    cmpq    %rbx, %r12                      # Compare loop index counter with total row count %rbx
    je      .L_cleanup                      # Jump to cleanup sequence if all rows have been processed
    
    movq    %r12, %rax                      # Copy current loop iteration index into %rax
    imulq   $SIZEOF_INPUT_ROW, %rax         # Calculate input data row byte offset position
    addq    %r13, %rax                      # Add host input buffer base address pointer
    movq    %rax, %rcx                      # Store computed input row memory pointer in %rcx
    
    movq    %r12, %rax                      # Copy current loop iteration index into %rax
    imulq   $SIZEOF_OUTPUT_ROW, %rax        # Calculate output data row byte offset position
    addq    %r14, %rax                      # Add host output buffer base address pointer
    movq    %rax, %rdx                      # Store computed output row memory pointer in %rdx

    cmpq    $1, %r15                        # Check if printing target destination is stdout terminal stream
    je      .L_print_terminal               # Jump to terminal formatted printing routine if true

    movsd   0(%rcx), %xmm0                  # Load input coefficient a1 into SSE vector register %xmm0
    movsd   8(%rcx), %xmm1                  # Load input coefficient a2 into SSE vector register %xmm1
    movsd   16(%rcx), %xmm2                 # Load input coefficient b1 into SSE vector register %xmm2
    movsd   24(%rcx), %xmm3                 # Load input coefficient b2 into SSE vector register %xmm3
    movsd   32(%rcx), %xmm4                 # Load input coefficient c1 into SSE vector register %xmm4
    movsd   40(%rcx), %xmm5                 # Load input coefficient c2 into SSE vector register %xmm5
    movsd   0(%rdx), %xmm6                  # Load root result component into %xmm6
    movsd   8(%rdx), %xmm7                  # Load root result component into %xmm7
    movsd   16(%rdx), %xmm8                 # Load root result component into %xmm8
    movsd   24(%rdx), %xmm9                 # Load root result component into %xmm9

    subq    $16, %rsp                       # Allocate stack alignment buffer space for dprintf arguments
    movsd   %xmm8, 0(%rsp)                  # Push root argument onto stack frame layout
    movsd   %xmm9, 8(%rsp)                  # Push root argument onto stack frame layout
    movq    %r15, %rdi                      # Move target file descriptor into %rdi
    leaq    csv_row_fmt(%rip), %rsi         # Load CSV row output formatting string pointer into %rsi
    movl    $8, %eax                        # Specify count of vector float arguments passed in registers
    call    dprintf@PLT                     # Write formatted CSV line directly to target file descriptor
    addq    $16, %rsp                       # Clean up temporary stack allocation space
    jmp     .L_loop_inc                     # Jump to iteration counter increment step

.L_print_terminal:
    leaq    res_fmt(%rip), %rdi             # Load human-readable console terminal result format string pointer
    movq    %r12, %rsi                      # Move current row index number into %rsi argument register
    movsd   0(%rdx), %xmm0                  # Load computed root real component value into %xmm0
    movsd   8(%rdx), %xmm1                  # Load computed root imaginary component value into %xmm1
    movsd   16(%rdx), %xmm2                 # Load second root real component value into %xmm2
    movsd   24(%rdx), %xmm3                 # Load second root imaginary component value into %xmm3
    movl    $4, %eax                        # Indicate 4 floating-point vector registers are populated
    call    printf@PLT                      # Print formatted root evaluation summary line to terminal

.L_loop_inc:
    incq    %r12                            # Increment loop row index counter value
    jmp     .L_print_loop                   # Return to start of printing loop iteration

.L_cleanup:
    movq    %rbp, %rsp                      # Restore original stack pointer from base pointer
    popq    %rbp                            # Restore saved frame pointer register value
    xorl    %edi, %edi                      # Clear exit status error code register (status = 0)
.L_exit:
    movq    $231, %rax                      # Set syscall number for exit_group (231)
    syscall                                 # Trigger final program termination system call

strcmp_local:
    xorl    %eax, %eax                      # Clear accumulator register for string comparison return value
.L_sloop:
    movb    (%rdi), %dl                     # Load byte from first comparison string pointer
    movb    (%rsi), %cl                     # Load byte from second comparison string pointer
    cmpb    %cl, %dl                        # Compare respective character bytes
    jne     .L_sdiff                        # Jump to difference handler if bytes do not match
    testb   %dl, %dl                        # Check if character byte is null string terminator
    jz      .L_sdone                        # Exit comparison loop successfully if end of strings met
    incq    %rdi                            # Advance first string pointer to next character
    incq    %rsi                            # Advance second string pointer to next character
    jmp     .L_sloop                        # Repeat string character comparison loop
.L_sdiff:
    sbbl    %eax, %eax                      # Compute character difference bit flag via borrow subtraction
    orl     $1, %eax                        # Ensure non-zero return code representing string inequality
.L_sdone:
    ret                                     # Return from local string comparison subroutine

.size _start, . - _start
.section .note.GNU-stack,"",@progbits
