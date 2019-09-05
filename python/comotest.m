% 
%myproblem = 'bi_sphere';

%Importing python module that we need :
py.importlib.import_module('mo');
% py.importlib.import_module('cma');
dim = py.int(10);
num_kernels = py.int(0);
sigma0 = py.float(0.2);
reference_point = py.numpy.array([1.1,1.1]);
dim = int64(dim);
x0 = py.numpy.array(ones(1,dim));
x_starts = py.list(repmat({x0},1,1));

%bounds = evalin('caller', '[0.5, inf]');
bounds = [0.5, inf];

if size(bounds, 1) > 1
    bounds = bounds';
end
lb = bounds(1);
rb = bounds(2);
if lb == -inf
    lb = -py.numpy.inf;
else
    lb = py.float(lb);
end
if rb == inf
    rb = py.numpy.inf;
else
    rb = py.float(rb);
end
newbounds = py.list({lb, rb});

nVar = 10;
num_offspring = py.int(floor(4+3*log(nVar)));
cmaes_opts = py.dict(struct('popsize', num_offspring, 'bounds', newbounds));
list_of_solvers = py.mo.get_cmas(x_starts, sigma0, cmaes_opts);

moes = py.mo.Sofomore(list_of_solvers,'reference_point', reference_point);
%while moes.stop() == 0
%while 0
for i =1:3
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

