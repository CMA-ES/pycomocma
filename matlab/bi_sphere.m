function [obj] = bi_sphere(x)
% simple two-sphere function
% note that this is an unconstrained benchmark problem
[N, nVar] = size(x);
obj = zeros(N, 2);

probsphere = py.problems.BiobjectiveConvexQuadraticProblem(pyargs('dim',py.int(nVar),'name','sphere'));
fun = probsphere.objective_functions();

for i = 1:N

sol = py.numpy.array(x(i,:));
obj(i, 1) = fun{1}(sol);
obj(i, 2) = fun{2}(sol);
end
end
