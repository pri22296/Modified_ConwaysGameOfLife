# Modified version of Conway's Game of Life

Usually `S[t] = CGOL(S[t-1])` where CGOL is the standard function of Conway's game of life,
and S is a binary matrix denoting if a cell is alive or dead.

Here it is modified to `S[t] = Min(1, CGOL(S[t-1]) + L[t-1]) * (I - D[t-1])`
where *L* and *D* are two matrices which denote Lifezone and a Deadzone.

All cells in a deadzone always remain dead and all cells in a alive zone always remain alive.
The matrices *L*, *D* can also change every iteration. In this implementation it's static but there is
no reason it can't change with time.
