%% MAU22602 / MAU34604 - Assignment 2 (Interpolation and Rootfinding)
% Name: Matthew Conway
% This sectioned MATLAB script is Live Script-ready.
% In MATLAB: Open this file and choose "Save As" -> .mlx if a .mlx submission is required.

clear; clc;

%% 1) Matrix Polynomials (written solutions)
% (a) Existence of p(A)=A^{-1} via Cayley-Hamilton
% Let chi_A(z)=det(zI-A)=z^n + c_{n-1}z^{n-1}+...+c_1 z + c_0.
% Cayley-Hamilton gives:
%   A^n + c_{n-1}A^{n-1} + ... + c_1 A + c_0 I = 0.
% Since A is invertible, c_0 = det(-A) = (-1)^n det(A) ~= 0.
% Rearranging:
%   c_0 I = -(A^n + c_{n-1}A^{n-1}+...+c_1A).
% Multiply by A^{-1}:
%   c_0 A^{-1} = -(A^{n-1}+c_{n-1}A^{n-2}+...+c_1 I).
% Therefore
%   A^{-1} = p(A)
% for a polynomial p of degree <= n-1.
%
% (b) Polynomial of a diagonalizable matrix
% If A = X*Lambda*X^{-1}, then A^k = X*Lambda^k*X^{-1} for all k>=0.
% For p(z)=sum_{k=0}^m a_k z^k,
%   p(A)=sum a_k A^k
%       =sum a_k (X*Lambda^k*X^{-1})
%       =X*(sum a_k Lambda^k)*X^{-1}
%       =X*p(Lambda)*X^{-1}.
% Because Lambda is diagonal, p(Lambda)=diag(p(lambda_1),...,p(lambda_n)).
%
% (c) Constructive proof via interpolation
% Let mu_1,...,mu_r be the distinct eigenvalues of A (all nonzero since A invertible).
% By interpolation existence, there is a polynomial p of degree <= r-1 with
%   p(mu_j)=1/mu_j,  j=1,...,r.
% Since p(Lambda)=diag(p(lambda_i)) and each lambda_i equals some mu_j,
%   p(Lambda)=diag(1/lambda_i)=Lambda^{-1}.
% Hence
%   p(A)=X*p(Lambda)*X^{-1}=X*Lambda^{-1}*X^{-1}=A^{-1}.

%% 2) Fixed Point Iteration
% Iteration: x_{i+1} = g(x_i) = 2x_i - c x_i^2, c ~= 0.
% (a) Fixed point x*=1/c:
%   g(1/c)=2/c - c*(1/c^2)=1/c.
%
% (b) Local convergence near x*=1/c:
%   g'(x)=2-2cx, so g'(1/c)=0. Therefore |g'(x)|<1 in a neighborhood of x*.
%   So the fixed-point iteration converges locally.
%
% (c) Explicit basin bound:
% Let e_i = x_i - 1/c. Then
%   e_{i+1} = x_{i+1} - 1/c = -c e_i^2,
% so |e_{i+1}| = |c| |e_i|^2.
% If |e_0| < 1/|c|, then |e_1| < |e_0| and in fact
%   |e_{i+1}|/|e_i| = |c||e_i| < 1,
% hence e_i -> 0 (quadratic once close).
% A valid bound is s = 1/|c|.

% Short numerical demo for part (2):
c = 2;
xstar = 1/c;
x0 = 0.60;
N = 8;
xs = zeros(1,N+1);
xs(1) = x0;
for k = 1:N
    xs(k+1) = 2*xs(k) - c*xs(k)^2;
end
fprintf('\nQ2 demo (c=%g):\n', c);
for k = 0:N
    fprintf('  k=%d, x_k=%.12f, |x_k-1/c|=%.3e\n', k, xs(k+1), abs(xs(k+1)-xstar));
end

%% 3) Rootfinding
% 3(a) Implemented local functions at end of file:
%   [alpha, n] = newton(f, df, x0, tol, maxit)
%   [alpha, n] = bisection(f, a, b, tol, maxit)
% Both terminate gracefully if not converged.

% 3(b) f(x)=x log(x)-1
f = @(x) x.*log(x) - 1;
df = @(x) log(x) + 1;

x0 = 2;
a = 0;
b = 100;
tol = 5e-7;     % enough for six correct decimals
maxit = 200;
alpha_true = 1.763222834351897;

[alpha_newton, n_newton] = newton(f, df, x0, tol, maxit);
[alpha_bisect, n_bisect] = bisection(f, a, b, tol, maxit);

fprintf('\nQ3 results:\n');
fprintf('  Newton   : alpha = %.15f, n = %d, |err| = %.3e\n', ...
    alpha_newton, n_newton, abs(alpha_newton-alpha_true));
fprintf('  Bisection: alpha = %.15f, n = %d, |err| = %.3e\n', ...
    alpha_bisect, n_bisect, abs(alpha_bisect-alpha_true));
fprintf('  6 d.p. check Newton   : %s\n', tfstr(abs(alpha_newton-alpha_true) < 5e-7));
fprintf('  6 d.p. check Bisection: %s\n', tfstr(abs(alpha_bisect-alpha_true) < 5e-7));

%% 4) Interpolation
% Data points:
xdata = [1, 2, 4, 5, 7, 8];
ydata = [0, 2, 12, 21, -1, -10];

% 4(a) Build p5(t) in three equivalent forms
n = numel(xdata);

% Power basis: p(t)=c0 + c1 t + ... + c5 t^5
V = zeros(n,n);
for j = 1:n
    V(:,j) = xdata'.^(j-1);
end
coeff_power = V \ ydata';
p_power = @(t) eval_power_basis(coeff_power, t);

% Lagrange basis
p_lagrange = @(t) eval_lagrange_basis(xdata, ydata, t);

% Newton basis via divided differences
coeff_newton = divided_difference_coeffs(xdata, ydata);
p_newton = @(t) eval_newton_basis(coeff_newton, xdata, t);

% 4(b) Evaluate at t=3 and t=6
tvals = [3, 6];
pp = p_power(tvals);
pl = p_lagrange(tvals);
pn = p_newton(tvals);

fprintf('\nQ4 results:\n');
for i = 1:numel(tvals)
    t = tvals(i);
    fprintf('  t=%g\n', t);
    fprintf('    power    : %.15f\n', pp(i));
    fprintf('    lagrange : %.15f\n', pl(i));
    fprintf('    newton   : %.15f\n', pn(i));
    fprintf('    |power-lagrange| = %.3e\n', abs(pp(i)-pl(i)));
    fprintf('    |power-newton|   = %.3e\n', abs(pp(i)-pn(i)));
    fprintf('    |lagrange-newton|= %.3e\n', abs(pl(i)-pn(i)));
end

%% Local functions
function [alpha, n] = newton(f, df, x0, tol, maxit)
    x = x0;
    for n = 1:maxit
        fx = f(x);
        dfx = df(x);
        if ~isfinite(fx) || ~isfinite(dfx) || dfx == 0
            fprintf('Newton stopped early at iteration %d (invalid derivative/value).\n', n);
            alpha = x;
            return;
        end

        xnew = x - fx/dfx;

        if ~isfinite(xnew)
            fprintf('Newton stopped early at iteration %d (non-finite iterate).\n', n);
            alpha = x;
            return;
        end

        if abs(xnew - x) < tol || abs(f(xnew)) < tol
            alpha = xnew;
            return;
        end

        x = xnew;
    end

    fprintf('Newton reached maxit=%d without meeting tolerance.\n', maxit);
    alpha = x;
end

function [alpha, n] = bisection(f, a, b, tol, maxit)
    % Gracefully handle an endpoint where f is not finite (e.g. x=0 for x log x).
    fa = f(a);
    fb = f(b);

    shiftCount = 0;
    while (~isfinite(fa) || ~isfinite(fb)) && shiftCount < 20
        if ~isfinite(fa)
            a = a + eps(max(1, abs(a)));
            fa = f(a);
        end
        if ~isfinite(fb)
            b = b - eps(max(1, abs(b)));
            fb = f(b);
        end
        shiftCount = shiftCount + 1;
    end

    if ~isfinite(fa) || ~isfinite(fb)
        fprintf('Bisection cannot start: non-finite endpoint values.\n');
        alpha = NaN;
        n = 0;
        return;
    end

    if sign(fa) == sign(fb)
        fprintf('Bisection cannot start: f(a) and f(b) have same sign.\n');
        alpha = NaN;
        n = 0;
        return;
    end

    for n = 1:maxit
        c = 0.5*(a+b);
        fc = f(c);

        if ~isfinite(fc)
            fprintf('Bisection stopped early at iteration %d (non-finite midpoint value).\n', n);
            alpha = c;
            return;
        end

        if abs(fc) < tol || 0.5*(b-a) < tol
            alpha = c;
            return;
        end

        if sign(fa) ~= sign(fc)
            b = c;
            fb = fc;
        else
            a = c;
            fa = fc;
        end
    end

    fprintf('Bisection reached maxit=%d without meeting tolerance.\n', maxit);
    alpha = 0.5*(a+b);
end

function y = eval_power_basis(coeff, t)
    y = zeros(size(t));
    for j = 1:numel(coeff)
        y = y + coeff(j) .* (t.^(j-1));
    end
end

function y = eval_lagrange_basis(xnodes, ynodes, t)
    m = numel(xnodes);
    y = zeros(size(t));

    for i = 1:m
        Li = ones(size(t));
        for j = 1:m
            if j ~= i
                Li = Li .* (t - xnodes(j)) / (xnodes(i) - xnodes(j));
            end
        end
        y = y + ynodes(i) * Li;
    end
end

function coeff = divided_difference_coeffs(xnodes, ynodes)
    m = numel(xnodes);
    dd = zeros(m,m);
    dd(:,1) = ynodes(:);

    for j = 2:m
        for i = 1:(m-j+1)
            dd(i,j) = (dd(i+1,j-1) - dd(i,j-1)) / (xnodes(i+j-1) - xnodes(i));
        end
    end

    coeff = dd(1,:).';
end

function y = eval_newton_basis(coeff, xnodes, t)
    m = numel(coeff);
    y = coeff(m) * ones(size(t));

    for k = m-1:-1:1
        y = coeff(k) + (t - xnodes(k)) .* y;
    end
end

function s = tfstr(tf)
    if tf
        s = 'YES';
    else
        s = 'NO';
    end
end
