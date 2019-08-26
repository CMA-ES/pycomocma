function [paretoFront, ...   % objectives
    paretoSet, ...           % parameters
    out] = COMOCMAES(...                 % struct with information
    problem, ...             % problem-string
    nObj, ...                % number of objectives
    xstart, ...              % initial sample point(s) (if only one point is given, the initial population will contain copies; len(xstart,1)=nVar (number of variables))
    insigma, ...             % initial step size(s)
    inopts, ...              % struct with options (optional)
    varargin)                % arguments passed to objective function

% OPTS = COMOCMAES returns default options.
% OPTS = COMOCMAES('defaults') returns default options quietly.
% OPTS = COMOCMAES('displayoptions') displays options.
% OPTS = COMOCMAES('defaults', OPTS) supplements options OPTS with default
% options.
%
% function call:
% [PARETOFRONT, PARETOSET[, OUT]] = COMOCMAES(PROBLEM, NOBJ, XSTART[, OPTS])
%
% Input arguments:
%  PROBLEM is a string function name like 'DTLZ1'. Each problem
%     takes as argument a column vector of variables together with the
%     number of objectives and a penalty factor and returns [objectives,
%     variables] (both column vectors). The feedback of variables can be
%     used to repair illegal values. The parameter 'penaltyfactor' can be
%     used to handle constraint violations.
%     Calling a problem with a matrix as first parameter interprets the
%     columns as individual solutions and computes the objective vectors
%     and the repaired variables for all solutions in parallel.
%  NOBJ gives the number of objectives of the multiobjective problem
%  XSTART indicates the initial sample points that will be used to
%     initialize the individual means of MO-CMA-ES's sample distributions.
%     The number of rows thereby gives the number of variables and the
%     number of columns the population size. If the number of columns does
%     not match the population size given in opts.nPop, the first column of
%     XSTART is used to initialize the mean vectors of all sample
%     distributions. Note that also a string can be given that will be
%     evaluated as MATLAB code such as 'rand(10, nPop)'.
%  OPTS (an optional argument) is a struct holding additional input
%     options. Valid field names and a short documentation can be
%     discovered by looking at the default options (type 'mocmaes'
%     without arguments, see above). Empty or missing fields in OPTS
%     invoke the default value, i.e. OPTS needs not to have all valid
%     field names.  Capitalization does not matter and unambiguous
%     abbreviations can be used for the field names. If a string is
%     given where a numerical value is needed, the string is evaluated
%     by eval, where
%     'nVar' expands to the problem dimension
%     'nObj' expands to the objectives dimension
%     'nPop' expands to the population size
%     'countEval' expands to the number of the recent evaluation
%     'nPV' expands to the number paretofronts
%
% Output:
%  PARETOFRONT is a matrix holding the objectives in rows. Each column
%     holds the objective vector of one solution.
%  PARETOSET is a matrix holding the parameters in rows. Each column holds
%     one solution.
%  OUT is a struct with additional information about the run. Most
%     important are probably OUT.nEval (number of evaluations performed),
%     OUT.stopflag (termination criterion of CMA-ES if active), and
%     OUT.termCrit (termination criterion of OCD if active).
%

% ----------- Importing python modules that we need --------------------
py.importlib.import_module('mo');
py.importlib.import_module('cma');
py.importlib.import_module('dill');

% ----------- Set Defaults for Options ---------------------------------
% options: general - these are evaluated once
defopts.nPop              = '10           % size of the population, which means here the number of kernels';
defopts.nOffspring        = '4+floor( 3*log(nVar) )           % number of offspring';
defopts.maxEval           = 'inf     % maximum number of evaluations';
defopts.refpoint          = '11.*ones(1, nObj) % reference point of the hypervolume';
defopts.bounds            = '[-inf, inf] % bounds manage the boundary constraints';
% bounds can be of size (1,2), or (2,1) or of type [lb; rb] where lb and rb
% respectfully represent the lower and upper bounds of size (1, nVar), or
% of type [lb, rb] where lb and rb respectfully represent the lower and
% upper bounds of size (nVar, 1).
defopts.OkResume          = '0 % resume former run';
defopts.maxiter           = 'inf % maximum number of iteration during a run';
defopts.number_asks        = '1 % the number of kernels from which we generate offspring simultaneously';
% In the algorithm, when we do 'moes.ask(number_asks)', then 'number_asks' kernels generate
% offspring, that they will pass to the 'tell' method, after all offspring
% are evaluated (in parallel). If opts.number_asks = 'all', then all
% kernels are asked during a call of the 'ask' method.


% ---------------------- Handling Input Parameters ----------------------

if nargin < 1 || isequal(problem, 'defaults') % pass default options
    if nargin < 5
        disp('Default options returned (type "help mocmaes" for help).');
    end
    paretoFront = defopts;
    if nargin > 5 % supplement second argument with default options
        paretoFront = getoptions(inopts, defopts);
    end
    return;
end

if isequal(problem, 'displayoptions')
    names = fieldnames(defopts);
    for name = names'
        disp([name{:} repmat(' ', 1, 20-length(name{:})) ': ''' defopts.(name{:}) '''']);
    end
    return;
end

if isempty(problem)
    error('Objective function not determined');
end
if ~ischar(problem)
    error('first argument ''problem'' must be a string');
end

% Compose options opts
if nargin < 5 || isempty(inopts) % no input options available
    opts = defopts;
else
    opts = getoptions(inopts, defopts);
end


% ------------------------ Initialization -------------------------------
%clc;

% get parameters for initialization

% initial means
if nargin < 3
    xstart = [];
end
if isempty(xstart)
    error('Initial search points and problem dimension not determined');
end
xstart = myeval(xstart);
maxiter = myeval(opts.maxiter);
maxEval = myeval(opts.maxEval);
number_asks = myeval(opts.number_asks);
Resume=myeval(opts.OkResume);
nVar = size(xstart,2);
nPop = size(xstart,1);
nPop_opts = myeval(opts.nPop); % in order to test for consistency
if nPop ~= nPop_opts
    xstart = repmat(xstart(1,:), nPop_opts, 1);
    nPop = nPop_opts;
end

% initial step sizes
if nargin < 4
    insigma = [];
end


if any(insigma) <= 0
    error('Initial step sizes cannot be <= 0.');
end
if size(insigma, 1) ~= nPop
    insigma = repmat(insigma(1, :), nPop, 1);
end


refpoint = myeval(opts.refpoint);
if size(refpoint, 1) ~= 1
    refpoint = refpoint';
end

reference_point = py.numpy.array(refpoint);
x_starts = py.list({});
sigma0 = py.list({});

for i = 1:nPop
    x_starts.append(py.numpy.array(xstart(i,:)));
    sigma0.append(py.float(insigma(i)));
end

bounds = myeval(opts.bounds);
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
num_offspring = py.int(myeval(opts.nOffspring));
cmaes_opts = py.dict(struct('popsize', num_offspring, 'bounds', newbounds));
list_of_solvers = py.mo.get_cmas(x_starts, sigma0, cmaes_opts);


% ------------------------ end Initialization -------------------------------

if ~Resume
    
    moes = py.mo.Sofomore(list_of_solvers,'reference_point', reference_point);
    
else
    %  try
    moes = py.dill.load(py.open('saved-mocma-object.pkl', 'rb'));
    % catch % initialize moes here
end
while moes.stop() == 0 && moes.countiter < maxiter && moes.countevals < maxEval
    %while 0
    X = moes.ask(py.int(number_asks)); %
    X_matlab = zeros(int64(py.len(X)), nVar);
    for i=1:size(X_matlab,1)
        X_matlab(i,:) = double(py.array.array('d',X{i}));
    end
    %    F_matlab = feval(problem, X_matlab, nObj, penaltyfactor); %problem takes more than one argument (nobj, penaltyfactor, ...)
    %   Constraints_matlab =
    F_matlab = feval(problem, X_matlab); %problem takes more than one argument (nobj, penaltyfactor, ...)
    F = py.list({});
    for i=1:size(X_matlab,1)
        F.append(py.list(F_matlab(i,:)));
    end
    
    moes.tell(X, F);
    % recupererer les parametres d'affichage pour le fracPareto
    moes.disp(50);
    drawnow; % to display immediately what is in disp()
    if Resume
    py.dill.dump(moes, py.open('saved-mocma-object.pkl', 'wb'))
    end
end


paretoFront = zeros(int64(py.len(moes.front)), nObj);
for i = 1:size(paretoFront,1)
    infront = moes.front(i);
    paretoFront(i,:) = double(py.array.array('d', infront{1}));
end

paretoSet = zeros(int64(py.len(moes.front)), nVar);
for i = 1:size(paretoFront,1)
    paretoSet(i,:) = double(py.array.array('d', moes.kernels{i}.incumbent));
end
out = struct(moes);



%%-------------------------------------------------------------------------
function opts=getoptions(inopts, defopts)
% OPTS = GETOPTIONS(INOPTS, DEFOPTS) handles an arbitrary number of
% optional arguments to a function. The given arguments are collected
% in the struct INOPTS.  GETOPTIONS matches INOPTS with a default
% options struct DEFOPTS and returns the merge OPTS.  Empty or missing
% fields in INOPTS invoke the default value.  Fieldnames in INOPTS can
% be abbreviated.
if nargin < 2 || isempty(defopts) % no default options available
    opts=inopts;
    return;
elseif isempty(inopts) % empty inopts invoke default options
    opts = defopts;
    return;
elseif ~isstruct(defopts) % handle a single option value
    if isempty(inopts)
        opts = defopts;
    elseif ~isstruct(inopts)
        opts = inopts;
    else
        error('Input options are a struct, while default options are not');
    end
    return;
elseif ~isstruct(inopts) % no valid input options
    error('The options need to be a struct or empty');
end

opts = defopts; % start from defopts
% if necessary overwrite opts fields by inopts values
defnames = fieldnames(defopts);
idxmatched = []; % indices of defopts that already matched
for name = fieldnames(inopts)'
    name = name{1}; % name of i-th inopts-field
    idx = strncmpi(defnames, name, length(name));
    if sum(idx) > 1
        error(['option "' name '" is not an unambigous abbreviation. ' ...
            'Use opts=RMFIELD(opts, ''' name, ...
            ''') to remove the field from the struct.']);
    end
    if sum(idx) == 1
        defname  = defnames{find(idx)};
        if ismember(find(idx), idxmatched)
            error(['input options match more than ones with "' ...
                defname '". ' ...
                'Use opts=RMFIELD(opts, ''' name, ...
                ''') to remove the field from the struct.']);
        end
        idxmatched = [idxmatched find(idx)];
        val = getfield(inopts, name);
        % next line can replace previous line from MATLAB version 6.5.0 on and in octave
        % val = inopts.(name);
        if isstruct(val) % valid syntax only from version 6.5.0
            opts = setfield(opts, defname, ...
                getoptions(val, getfield(defopts, defname)));
        elseif isstruct(getfield(defopts, defname))
            % next three lines can replace previous three lines from MATLAB
            % version 6.5.0 on
            %   opts.(defname) = ...
            %      getoptions(val, defopts.(defname));
            % elseif isstruct(defopts.(defname))
            warning(['option "' name '" disregarded (must be struct)']);
        elseif ~isempty(val) % empty value: do nothing, i.e. stick to default
            opts = setfield(opts, defnames{find(idx)}, val);
            % next line can replace previous line from MATLAB version 6.5.0 on
            % opts.(defname) = inopts.(name);
        end
    else
        warning(['option "' name '" disregarded (unknown field name)']);
    end
end

%%-------------------------------------------------------------------------
%%-------------------------------------------------------------------------
function res=myeval(s)
if ischar(s)
    if strncmpi(s, 'yes', 3) || strncmpi(s, 'on', 2) ...
            || strncmpi(s, 'true', 4) || strncmp(s, '1 ', 2)
        res = 1;
    elseif strncmpi(s, 'no', 2) || strncmpi(s, 'off', 3) ...
            || strncmpi(s, 'false', 5) || strncmp(s, '0 ', 2)
        res = 0;
    else
        try res = evalin('caller', s); catch
            error(['String value "' s '" cannot be evaluated']);
        end
        try res ~= 0; catch
            error(['String value "' s '" cannot be evaluated reasonably']);
        end
    end
else
    res = s;
end

%%-------------------------------------------------------------------------


function feasx = feasible(x, bounds)
% in x: one column per solution, variables in rows
% in bounds: column1: lower bound; column2: upper bound; rows=variables
feasx = min(max(x, repmat(bounds(:,1), 1, size(x,2))), ...
    repmat(bounds(:,2), 1, size(x,2)));


function penalizedf = penaltyterm(x, feasx, penaltyfactor)
% x,feasx: one column per solution, variables in rows
penalizedf = penaltyfactor .* sum((x - feasx).^2);


