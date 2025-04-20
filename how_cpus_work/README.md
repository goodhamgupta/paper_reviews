# What Every Programmer Should Know about How CPUs Work • Matt Godbolt • GOTO 2024

Beautiful presentation regarding the internals of CPUs and how they execute code.
The talk primarily focuses on the x86 architecutre, and shows performance of simple programs on ONE CPU.
Tools like `perf` and `Instruments` are used to visualize the performance of the code.

Branch prediction is the key sauce of performance in modern CPUs, and is a heavily guarded secret by CPU manufacturers. However, they can be reverse engineered.

The compiler is SUPER SMART, and can optimize code in ways that are not obvious to the programmer. The compiler can reorder instructions, inline functions, and even eliminate dead code.
