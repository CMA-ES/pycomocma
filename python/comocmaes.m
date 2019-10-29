function [paretoFront, ...   % objectives
    paretoSet, ...           % parameters
    out, opts] = COMOCMAES(...                 % struct with information
    problem, ...             % problem-string
    nObj, ...                % number of objectives
    xstart, ...              % initial sample point(s) (if only one point is given, the initial population will contain copies; len(xstart,1)=nVar (number of variables))
    insigma, ...             % initial step size(s)
    inopts)              % struct with options (optional)

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
%  OUT is a struct with additional information about the run.
%



% ----------- Set Defaults for Options ---------------------------------
% options: general - these are evaluated once
defopts.nPop              = '10           % size of the population, which means here the number of kernels';
defopts.nOffspring        = '(4+floor( 3*log(nVar) )) * 10           % total number of offspring';
defopts.popsize           = '4+floor( 3*log(nVar) )           % number of offspring per kernel';
defopts.maxEval           = 'inf     % maximum number of evaluations';
defopts.refpoint          = '(1 + nVar).*ones(1, nObj) % reference point of the hypervolume';
defopts.bounds            = '[-inf, inf] % bounds manage the boundary constraints';
% bounds can be of size (1,2), or (2,1) or of type [lb; rb] where lb and rb
% respectfully represent the lower and upper bounds of size (1, nVar), or
% of type [lb, rb] where lb and rb respectfully represent the lower and
% upper bounds of size (nVar, 1).
defopts.okresume          = 'False % resume former run';
defopts.resumefile        = '';
defopts.maxiter           = 'inf % maximum number of iteration during a run';
defopts.number_asks       = 'py.str("all") % the number of kernels from which we generate offspring simultaneously';
% In the algorithm, when we do 'moes.ask(number_asks)', then 'number_asks' kernels generate
% offspring, that they will pass to the 'tell' method, after all offspring
% are evaluated (in parallel). If opts.number_asks = 'all', then all
% kernels are asked during a call of the 'ask' method.
defopts.tolx              = '1e-6 % tolerence in x for stopping criterion';
defopts.frac_inactive         = '1 % fraction of inactive kernel for stopping';
defopts.logger            = '1 % if 1, we log data to file. And if 0, we do not';
defopts.elitist           = '0 % 3 possibilities: 0 , 1 , 2'; 
% if 0: non-elitist mode, if 1: elitist mode, and 2  for 'init': only
% the start is elitist.
defopts.verb_disp         = '100 % display results each verb_disp iterations';
defopts.display           = 'on            % display some things during the run and graphics';
defopts.showWaitbar       = 'on  % FHU display waitbar if closed during process stop optimization';
defopts.abscissa          = '0 % if 1, the x_axis represents count evals, and if 0, it shows the iterations ';

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
% if ~(ischar(problem) 
%     error('first argument ''problem'' must be a string or an handle function');
% end

% Compose options opts
if nargin < 5 || isempty(inopts) % no input options available
    opts = defopts;
else
    opts = getoptions(inopts, defopts);
end

% ----------- Importing python modules that we need --------------------
% opts.RepPy='Y:\_GEO\COMMUN\LOGICIELS_FHU\Optimisarion\COMO-CMAES-Cheikh\comocmaes\src';
% py.importlib.import_module(fullfile(opts.RepPy,'como'));
% py.importlib.import_module(fullfile(opts.RepPy,'cma'));
py.importlib.import_module('como');
py.importlib.import_module('cma');

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
resume = myeval(opts.okresume);
resumefile =opts.resumefile;
%resumefile = py.str(resumefile); % in case it's a matlab char
logger = myeval(opts.logger);
display = myeval(opts.display);
abscissa = myeval(opts.abscissa);
nVar = size(xstart,2);
nPop = size(xstart,1);
tolx = myeval(opts.tolx);
frac_inactive=myeval(opts.frac_inactive);
verb_disp = myeval(opts.verb_disp);
displayWaitbar    = myeval(opts.showWaitbar); % FHU display waitbar
elitist = myeval(opts.elitist);
if elitist == 0
    elitist = py.False;
elseif elitist == 2
    elitist = py.str('init');
else
    elitist = py.True;
end
nPop_opts = myeval(opts.nPop); % in order to test for consistency
if nPop ~= nPop_opts
    xstart = repmat(xstart(1,:), nPop_opts, 1);
    nPop = nPop_opts;
end
number_asks = myeval(opts.number_asks);
if isa(number_asks, 'double') || isa(number_asks, 'int64') || isa(number_asks, 'int32')
    number_asks = py.int(number_asks);
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
if size(bounds, 1) ~= 2
    bounds = bounds';
end
lb = bounds(1, :);
rb = bounds(2, :);

newbounds = py.list({lb, rb});
popsize = py.int(myeval(opts.popsize));
%TODO: authorize the possibility for different options to different cma-es
cmaes_opts = py.dict(struct('popsize', popsize, 'bounds', newbounds, 'tolx', tolx, 'CMA_elitist', elitist));
list_of_solvers = py.como.get_cmas(x_starts, sigma0, pyargs('inopts', cmaes_opts));

if display && logger

    figure(44444);
    h(1)=subplot(2,2,1);%#TODO: FHU subplot initialization to avoid trouble with waitbar
    h(2)=subplot(2,2,2);
    h(3)=subplot(2,2,3);
    h(4)=subplot(2,2,4);
    %         h(5)=subplot(2,3,5);
    %         h(6)=subplot(2,3,6);
    hold off;
    
    
end

% ------------------------ end Initialization -------------------------------

if ~resume
    
    moes = py.como.Sofomore(list_of_solvers,'reference_point', reference_point);
    
else
    try
        moes = py.pickle.load(py.open(py.str(resumefile) + py.str('.pkl'), 'rb'));
        %TODO: a way to modify the options of moes here.
    catch % when the file doesn't exist
        %(for example 0 iteration has been made).
        moes = py.como.Sofomore(list_of_solvers,'reference_point', reference_point);
    end
    
    
end

stopflag = {};
stopByUser=false;
if displayWaitbar
    try
        hw= waitbar(int64(moes.countiter)/maxiter,WaitbarString,'Name','COMOCMAES');
    catch
        
        hw= waitbar(int64(moes.countiter)/maxiter,'from resume','Name','COMOCMAES');
    end
end
%% boucle de calage
inactive=0;
while moes.stop() == 0 && moes.countiter < maxiter && moes.countevals < maxEval && ~stopByUser && inactive<frac_inactive 
    
    X = moes.ask(number_asks); %
    X_matlab = zeros(int64(py.len(X)), nVar);
    for i=1:size(X_matlab,1)
        X_matlab(i,:) = double(py.array.array('d',X{i}));
    end
    f_values_and_constraints = feval(problem, X_matlab);
    F_matlab = f_values_and_constraints(:, 1:nObj);
    C_matlab = f_values_and_constraints(:,nObj+1:end);
    C = py.list({});
    if size(C_matlab, 2) ~= 0 % we have constraints
        for i=1:size(C_matlab,2)
            C.append(py.list(C_matlab(:,i)'));
            %        C.append(py.list(C_matlab(i,:)))
        end
    end
    F = py.list({});
    for i=1:size(X_matlab,1)
        F.append(py.list(F_matlab(i,:)));
        %        C.append(py.list(C_matlab(i,:)))
    end
    
    moes.tell(X, F, py.list(C));

    moes.disp(verb_disp);
    drawnow; % to display immediately what is in disp()
    
    
    if logger
        moes.logger.add()
    end
    
    if strcmp(resumefile, '') == 0
        py.pickle.dump(moes, py.open(py.str(resumefile) + py.str('.pkl'), 'wb'))
        % TODO: expose the name of the saved file in the options
        
    end
    
    
    if display && logger && mod(int64(moes.countiter), verb_disp) == 0
        inactive=myplot(moes, abscissa, nObj, nVar, opts, hw);%, refpoint);
        
    elseif logger && mod(int64(moes.countiter), verb_disp) == 0
        filenames_ratio = py.list({moes.logger.name_prefix + py.str('ratio_inactive_kernels.dat'),...
            moes.logger.name_prefix + py.str('ratio_nondom_incumb.dat'), moes.logger.name_prefix + py.str('ratio_nondom_offsp_incumb.dat')});
        tuple_ratio = moes.logger.load(filenames_ratio);
        res_ratio = tuple_ratio{3};
        inactive = max(double(py.array.array('d', res_ratio{1})));
        
    end
    displayWaitbar = myeval(opts.showWaitbar);
    if displayWaitbar && moes.countiter > 2
        max_max_stds = double(moes.max_max_stds);
        WaitbarString=['It',num2str(int64(moes.countiter)), ' HVmax=',num2str(double(py.float(moes.best_hypervolume_pareto_front)), '%.3e'),' max(max stds)=', num2str(max_max_stds, '%.3e')];
        
        maxiter = myeval(opts.maxiter);
        if ishandle(hw)
            waitbar(double(int64(moes.countiter))/maxiter,hw,WaitbarString,'Name','COMOCMAES');
        else % waitbar closed-> stop
            stopByUser=true;
            stopflag={stopflag{:},'Stop by user'};
        end
    end
    
    
end

% last display (end of while loop):
if display
    myplot(moes, abscissa, nObj, nVar, opts, hw);
end


paretoFront = zeros(int64(py.len(moes.pareto_front)), nObj);
for i = 1:size(paretoFront,1)
    infront = moes.pareto_front{i};
    paretoFront(i,:) = double(py.array.array('d', infront));
end

paretoSet = zeros(int64(py.len(moes.pareto_front)), nVar);
for i = 1:size(paretoFront,1)
    paretoSet(i,:) = double(py.array.array('d', moes.kernels{i}.incumbent));
end
out = struct();

archive = zeros(int64(py.len(moes.archive)), nObj);
for i = 1:size(archive,1)
    infront = moes.archive{i};
    archive(i,:) = double(py.array.array('d', infront));
end
out.archive = archive;
out.termination_status = moes.termination_status;
out.num_kernels = int64(moes.num_kernels);
out.stop = moes.stop();
out.countiter = int64(moes.countiter);
out.countevals = double(moes.countevals);
out.best_hypervolume_pareto_front = double(py.float(moes.best_hypervolume_pareto_front));
out.hypervolume_pareto_front = double(py.float(moes.pareto_front.hypervolume));
out.hypervolume_archive = double(py.float(moes.archive.hypervolume));
if ishandle(hw)
    close(hw);
end
end
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
end


function max_inactive=myplot(moes, abscissa, nObj, nVar, opts, hw, refpoint)
filenames_ratio = py.list({moes.logger.name_prefix + py.str('ratio_inactive_kernels.dat'),...
    moes.logger.name_prefix + py.str('ratio_nondom_incumb.dat'), moes.logger.name_prefix + py.str('ratio_nondom_offsp_incumb.dat')});
tuple_ratio = moes.logger.load(filenames_ratio);
iter_ratio = double(py.array.array('d', tuple_ratio{1}));
countevals_ratio = double(py.array.array('d', tuple_ratio{2}));
res_ratio = tuple_ratio{3};

filenames_hypervolume = py.list({moes.logger.name_prefix + py.str('hypervolume.dat'),...
    moes.logger.name_prefix + py.str('hypervolume_archive.dat'), moes.logger.name_prefix + py.str('len_archive.dat')});
tuple_hypervolume = moes.logger.load(filenames_hypervolume);
iter_hypervolume = double(py.array.array('d', tuple_hypervolume{1}));
countevals_hypervolume = double(py.array.array('d', tuple_hypervolume{2}));
res_hypervolume = tuple_hypervolume{3};

filenames_median = py.list({moes.logger.name_prefix + py.str('median_sigmas.dat'),...
    moes.logger.name_prefix + py.str('median_min_stds.dat'), moes.logger.name_prefix + py.str('median_max_stds.dat')});
tuple_median = moes.logger.load(filenames_median);
iter_median = double(py.array.array('d', tuple_median{1}));
countevals_median= double(py.array.array('d', tuple_median{2}));
res_median = tuple_median{3};

filenames_median_stds = py.list({moes.logger.name_prefix + py.str('median_stds.dat')});
tuple_median_stds = moes.logger.load(filenames_median_stds);
iter_median_stds = double(py.array.array('d', tuple_median_stds{1}));
countevals_median_stds= double(py.array.array('d', tuple_median_stds{2}));
res_median_stds_python = py.list(tuple_median_stds{3}{1});
res_median_stds = zeros(size(iter_median_stds, 2), nVar);
for i=1:size(res_median_stds, 1)
    res_median_stds(i,:) = double(py.array.array('d',res_median_stds_python{i}));
end

if abscissa
    x_axis_hypervolume = countevals_hypervolume;
    x_axis_median = countevals_median;
    x_axis_ratio = countevals_ratio;
    x_axis_median_stds = countevals_median_stds;
else
    x_axis_hypervolume = iter_hypervolume;
    x_axis_median = iter_median;
    x_axis_ratio = iter_ratio;
    
    x_axis_median_stds = iter_median_stds;
    %         x_axis_median_stds = countevals_median_stds;
end

figure(44444);
h(1)=subplot(2,2,1);%#TODO: FHU subplot initialization to avoid trouble with waitbar
h(2)=subplot(2,2,2);
h(3)=subplot(2,2,3);
h(4)=subplot(2,2,4);
%fixedpop = population; % testing
linestyle = '--';
set(0, 'CurrentFigure', 44444);

%%%%%%%%%%%%%%%%
%             subplot(2,3,1);
axes(h(1)); %TODO FHU: replace subplot by axes
%%%%%%%%%%%%%%%%
% evolution of HV:
cla;
hold off;
HV = double(py.array.array('d', res_hypervolume{1}));
HV_max = double(py.float(moes.best_hypervolume_pareto_front));
sigmas = double(py.array.array('d', res_median{1}));
min_stds = double(py.array.array('d', res_median{2}));
max_stds = double(py.array.array('d', res_median{3}));


semilogy(x_axis_hypervolume, HV_max - HV, 'LineStyle', linestyle, ...
    'Color', 'red');
hold on;
if moes.isarchive ~= 0
    HV_archive = double(py.array.array('d', res_hypervolume{2}));
    HV_archive_max = HV_archive(end);
    length_archive = double(py.array.array('d', res_hypervolume{3}));
    inverse_length_archive = 1./length_archive;
    semilogy(x_axis_hypervolume, HV_archive_max - HV_archive, 'LineStyle', linestyle, ...
        'Color', 'blue');
    hold on
    semilogy(x_axis_hypervolume, inverse_length_archive, 'LineStyle', linestyle, ...
        'Color', 'cyan');
end

%         hold on
%         semilogy(x_axis_median, sigmas, 'LineStyle', linestyle, ...
%             'Color', 'k');
%         hold on
%         semilogy(x_axis_median, min_stds, 'LineStyle', linestyle, ...
%             'Color', 'green');
%         hold on
%         semilogy(x_axis_median, max_stds, 'LineStyle', linestyle, ...
%             'Color', 'green');
%                 hold on
%             legend('HV_{max} - HV', 'HV_{archive_{max}} - HV_{archive}', 'inverse length archive', 'median sigmas', 'median min stds', 'median max stds');
title('HV_{max}-HV (red,blue), inverse-length-archive(cyan) ');
if abscissa
    xlabel('Count evals');
else
    xlabel('Niter');
end
%text(0,100,sprintf('HV = %e', max(HVtotal)));
ax = axis;
text(ax(1), 10^(log10(ax(3))+0.05*(log10(ax(4))-log10(ax(3)))), ...
    [ ' HV=' num2str(double(py.float(moes.best_hypervolume_pareto_front)), '%.15g')]);


grid on;

%%%%%%%%%%%%%%%%
%             subplot(2,3,2);
axes(h(2)); %TODO FHU: replace subplot by axes
%%%%%%%%%%%%%%%%
% population in objective space over time:
cla;
hold off;

currentFront = zeros(int64(py.len(moes.pareto_front)), nObj);
for i = 1:size(currentFront,1)
    infront = moes.pareto_front{i};
    currentFront(i,:) = double(py.array.array('d', infront));
end
% need to plot all kernels in objective space
% All kernels in objective space
All_kernels = zeros(int64(py.len(moes.kernels)), nObj);
for i = 1:size(All_kernels,1)
    kernel = moes.kernels{i};
    infront = kernel.objective_values;
    All_kernels(i,:) = double(py.array.array('d', infront));
end


if moes.isarchive ~= 0
    currentArchive = zeros(int64(py.len(moes.archive)), nObj);
    for i = 1:size(currentArchive,1)
        infront = moes.archive{i};
        currentArchive(i,:) = double(py.array.array('d', infront));
    end
end

if nObj == 3 % 3 objectives
    
    if moes.isarchive ~= 0
        plot3(currentArchive(:,1), currentArchive(:,2), currentArchive(:,3),'.', 'Color', 'blue');
        xlabel('f_1');
        ylabel('f_2');
        zlabel('f_3');
        MinAbciss=min(currentArchive);
        hold on;
    else
        MinAbciss=min(currentFront);
    end
    plot3(All_kernels(:,1), All_kernels(:,2), All_kernels(:,3),'o', 'Color', 'green');hold on
    plot3(currentFront(:,1), currentFront(:,2), currentFront(:,3),'o', 'Color', 'red');
    xlabel('f_1');
    ylabel('f_2');
    zlabel('f_3');
    if nargin>6
        xlim([MinAbciss(1),refpoint(1)]);
        ylim([MinAbciss(2),refpoint(2)]);
        zlim([MinAbciss(3),refpoint(3)]);
    end
else  % 2 objectives
    
    
    if moes.isarchive ~= 0
        plot(currentArchive(:,1), currentArchive(:,2),'.', 'Color', 'blue');
        xlabel('f_1');
        ylabel('f_2');
        hold on;
        MinAbciss=min(currentArchive);
    else
        MinAbciss=min(currentFront);
    end
    plot(All_kernels(:,1), All_kernels(:,2),'o', 'Color', 'green');hold on
    plot(currentFront(:,1), currentFront(:,2),'o', 'Color', 'red');
    xlabel('f_1');
    ylabel('f_2');
    if nargin>6
        xlim([MinAbciss(1),refpoint(1)]);
        ylim([MinAbciss(2),refpoint(2)]);
    end
end

title('objective space: archive (blue), kernels(red)');

%         legend('estimated front','archive');
grid on;

%%%%%%%%%%%%%%%%
%             subplot(2,3,3);
axes(h(3)); %TODO FHU: replace subplot by axes
cla;
% axis([-0.01, 1.01]);
%%%%%%%%%%%%%%%%
% population in decision space (projection) over time:
hold off;

inactive = double(py.array.array('d', res_ratio{1}));
max_inactive=max(inactive);
nondom_incumbent = double(py.array.array('d', res_ratio{2}));
%         first_quartile_nondom_offspring_incumbent = double(py.array.array('d', res_ratio{3}));
median_nondom_offspring_incumbent = double(py.array.array('d', res_ratio{4}));
%         last_quartile_nondom_offspring_incumbent = double(py.array.array('d', res_ratio{5}));

plot(x_axis_ratio, inactive, '-', 'Color', 'red');
hold on;
plot(x_axis_ratio, nondom_incumbent, '-', 'Color', 'blue');
hold on;
%             plot(x_axis_ratio, first_quartile_nondom_offspring_incumbent, '-', 'Color', 'green');
%         hold on;
plot(x_axis_ratio, median_nondom_offspring_incumbent, '-', 'Color', 'k');
%             hold on;
%             plot(x_axis_ratio, last_quartile_nondom_offspring_incumbent, '-', 'Color', 'green');

if abscissa
    xlabel('Count evals');
else
    xlabel('Niter');
end
ylabel('Ratios');
title('Fracpareto(Kernels:blue,offspring:black) and inactive kernels(red)');
%         legend('inactive kernels','non-dominated incumbents','1st quartile non-dom incumbent + offspring',...
%             'median non-dom incumbent + offspring', 'last quartile non-dom incumbent + offspring');
grid on;

%%%%%%%%%%%%%%%%
%             subplot(2,3,5);
axes(h(4)); %TODO FHU: replace subplot by axes
%%%%%%%%%%%%%%%%
% axis ratios of all covariances:
cla;
hold off;
semilogy(x_axis_median_stds, res_median_stds);
hold on;
tol = myeval(opts.tolx);
tol = tol / double(moes.kernels{1}.sigma0);
semilogy(x_axis_median_stds, tol * ones(size(x_axis_median_stds)));

if abscissa
    xlabel('Count evals');
else
    xlabel('Niter');
end
title('(sorted) median standard deviations');
grid on

hold off;
drawnow;



end % of plotting
%%-------------------------------------------------------------------------

