import sympy as sp

# Given sequence data
data = [2, 5, 12, 29]

# Step 1: Guess recurrence a_n = c1*a_(n-1) + c2*a_(n-2)
c1, c2 = sp.symbols("c1 c2", real=True)
eqs = []

for n in range(2, len(data)):
    eqs.append(sp.Eq(data[n], c1*data[n-1] + c2*data[n-2]))

sol = sp.solve(eqs, (c1, c2))
print("Reccurence:", sol)

# Step 2: Solve characteristic equation
c1_val, c2_val = sol[c1], sol[c2]
x = sp.symbols("x")
char_eq = sp.Eq(x**2 - c1_val*x - c2_val, 0)
roots = sp.solve(char_eq, x)
print("Roots:", roots)

# Step 3: General form
r1, r2 = roots
α,β = sp.symbols("α β")
n = sp.symbols("n")

general = α*r1**n + β*r2**n

# Solve α, β from initial conditions
sol_ab = sp.solve([
    sp.Eq(general.subs(n, 0), data[0]),
    sp.Eq(general.subs(n, 1), data[1])
], [α, β])

print("Closed-form a_n = α*r1^n + β*r2^n")
print(sol_ab)


