%myproblem = 'bi_sphere';

%Importing python module that we need :
py.importlib.import_module('mo');
% py.importlib.import_module('cma');
dim = py.int(10);
num_kernels = py.int(5);
sigma0 = py.float(0.2);
reference_point = py.numpy.array([1.1,1.1]);
dim = int64(dim);
x0 = py.numpy.array(ones(1,dim));
x_starts = py.list(repmat({x0},1,5));
list_of_solvers = py.mo.get_cma(x_starts, sigma0);

moes = py.mo.Sofomore(list_of_solvers,'reference_point', reference_point);

X = moes.ask("all");

for i = 1:int64(py.len(moes.asked_indices))
    kernel = moes.kernels{py.int(i-1)};
    kernel.objective_values = py.list(bi_sphere(double(py.array.array('d',kernel.mean))));
end
X_matlab = zeros(int64(py.len(X)), dim);
for i=1:size(X_matlab,1)
   X_matlab(i,:) = double(py.array.array('d',X{i}));
end
 
F_matlab = bi_sphere(X_matlab);
F = py.list({});
for i=1:size(X_matlab,1)
    F.append(py.list(F_matlab(i,:)));
end

moes.tell(X, F);

%while ~int64(py.len(moes.stop()))
while moes.stop() == 0
    for i=1:int64(py.len(moes.asked_indices))
        kernel = moes.kernels{i};
        kernel.objective_values = py.list(bi_sphere(double(py.array.array('d',kernel.mean))));
    end
    X = moes.ask();
    X_matlab = zeros(int64(py.len(X)), dim);
    for i=1:size(X_matlab,1)
        X_matlab(i,:) = double(py.array.array('d',X{i}));
    end
    F_matlab = bi_sphere(X_matlab);
    F = py.list({});
    for i=1:size(X_matlab,1)
        F.append(py.list(F_matlab(i,:)));
    end
    moes.tell(X, F)
end

