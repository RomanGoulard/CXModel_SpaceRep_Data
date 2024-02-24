flush
warning('off','all')

advance_bar = waitbar(0,'Data loading...');

set(0,'defaultAxesFontSize',4)

%% Folder selection
folder_simulation = uigetdir('D:\PyCharmProjects\InsectNeuroNano_Results\','Select results folder');
folder_results = fullfile(folder_simulation, 'Result_files');

cd(folder_simulation)

gainvalues_search = dir('Gains_list.txt');
gainvalues_bool = logical(~isempty(gainvalues_search));

if gainvalues_bool
    gainvalues = readtable('Gains_list.txt');
end

LocParams_rnd = floor(gainvalues.Loc*10)/10 + 0.05;
GloParams_rnd = floor(gainvalues.Glo*10)/10 + 0.05;

Xvec_gainparam = min(LocParams_rnd):0.1:max(LocParams_rnd);
Yvec_gainparam = min(GloParams_rnd):0.1:max(GloParams_rnd);

Result_global.ParamMap_count = zeros(length(Xvec_gainparam),length(Yvec_gainparam));
Result_global.ParamMap_count_time = zeros(length(Xvec_gainparam),length(Yvec_gainparam));
Result_global.ParamMap_successrate = zeros(length(Xvec_gainparam),length(Yvec_gainparam));
Result_global.ParamMap_successtime = zeros(length(Xvec_gainparam),length(Yvec_gainparam));

Result_global.VM_ParamMap_count = zeros(length(Xvec_gainparam),length(Yvec_gainparam));
Result_global.VM_ParamMap_dist2aim = zeros(length(Xvec_gainparam),length(Yvec_gainparam));
Result_global.VM_HeatmapReturn = [];

if ~isempty(dir('World_items.csv'))
    ObjWorld = readtable('World_items.csv');
    World_log = 1;
else
    ObjWorld = NaN;
    World_log = 0;
end

cd(folder_results)

folder_content = dir(folder_results);

%%
hDc2PFL = zeros(16,16);
for in1 = 1:16
    for in2 = 1:16
        if (in1 <= 8) && ((in1-in2)==-8)
            hDc2PFL(in1,in2) = 1.0;
        elseif (in1 > 8) && ((in1-in2)==8)
            hDc2PFL(in1,in2) = 1.0;
        end
    end
end

%% Files scan
CSVfile_list = dir('*.csv');
CSVfile_list_foodsources = dir('*FoodSources.csv');
CSVfile_list_XY = dir('*Results_XY.csv');

CSVfile_list_pol = dir('*CIN.csv');
CSVfile_list_epg = dir('*EPG.csv');
CSVfile_list_pen = dir('*PEN.csv');
CSVfile_list_peg = dir('*PEG.csv');
CSVfile_list_d7 = dir('*D7.csv');
CSVfile_list_no = dir('*NOD.csv');
CSVfile_list_pfn = dir('*PFN.csv');
CSVfile_list_pinfn = dir('*PinFN.csv');
CSVfile_list_hdc = dir('*hDc.csv');
CSVfile_list_hindc = dir('*hinDc.csv');
CSVfile_list_pfl = dir('*PFL.csv');
CSVfile_list_lal = dir('*LAL.csv');
CSVfile_list_fbt_hd_vecmemo = dir('*FBt_HD_memo.csv');
CSVfile_list_fbt_pi_vecmemo = dir('*FBt_PI_memo.csv');

name_files_foodsources = {CSVfile_list_foodsources.name};
name_files_XY = {CSVfile_list_XY.name};

name_files_pol = {CSVfile_list_pol.name};
name_files_epg = {CSVfile_list_epg.name};
name_files_pen = {CSVfile_list_pen.name};
name_files_peg = {CSVfile_list_peg.name};
name_files_d7 = {CSVfile_list_d7.name};
name_files_no = {CSVfile_list_no.name};
name_files_pfn = {CSVfile_list_pfn.name};
name_files_pinfn = {CSVfile_list_pinfn.name};
name_files_hdc = {CSVfile_list_hdc.name};
name_files_hindc = {CSVfile_list_hindc.name};
name_files_pfl = {CSVfile_list_pfl.name};
name_files_lal = {CSVfile_list_lal.name};
name_files_fbt_hd_vecmemo = {CSVfile_list_fbt_hd_vecmemo.name};
name_files_fbt_pi_vecmemo = {CSVfile_list_fbt_pi_vecmemo.name};


%% Experiments count
name_files = {CSVfile_list_epg.name};
numexp_list = nan(1, length(name_files));
for ifile = 1:length(name_files)
    num_exp = regexp(name_files{ifile},'\d*','Match');
    num_exp = str2num(num_exp{1});
    numexp_list(ifile) = num_exp;
end

NumExp_list = unique(numexp_list);

%% Main figures generation

fig0_CXinputs.mainfig = figure('position', [50 60 800 500], 'renderer', 'painters');
fig0_CXinputs.tabgroup = uitabgroup(fig0_CXinputs.mainfig); % tabgroup

fig1_XYplot.mainfig = figure('position', [125 70 1000 500], 'renderer', 'painters');
fig1_XYplot.tabgroup = uitabgroup(fig1_XYplot.mainfig); % tabgroup

fig2_compass.mainfig = figure('position', [200 80 800 500], 'renderer', 'painters');
fig2_compass.tabgroup = uitabgroup(fig2_compass.mainfig); % tabgroup
        
%% Data extraction
for ifile = 1:length(NumExp_list)
    waitbar((ifile-1)/length(NumExp_list),advance_bar,['Data processing... (' num2str(ifile) '/' num2str(length(NumExp_list)) ')']);
    
    [~,Xgainparam] = min(abs(Xvec_gainparam-LocParams_rnd(ifile)));
    [~,Ygainparam] = min(abs(Yvec_gainparam-GloParams_rnd(ifile)));
    Result_global.ParamMap_count(Xgainparam,Ygainparam) = Result_global.ParamMap_count(Xgainparam,Ygainparam)+1;
    
    if ifile <= length(name_files_XY)
    %% Data extraction
        if ~isempty(name_files_foodsources)
            data_foodsources = csvread(name_files_foodsources{ifile});
            if ~isnan(data_foodsources)
                rot_rho_foodsources = sqrt(data_foodsources(:,1).^2 + data_foodsources(:,2).^2);
                rot_sig_foodsources = atan2d(-data_foodsources(:,2), data_foodsources(:,1));
                % Corrected data
                X_foodsources = cosd(rot_sig_foodsources) .* rot_rho_foodsources;
                Y_foodsources = sind(rot_sig_foodsources) .* rot_rho_foodsources;
            end
        else
            data_foodsources = NaN;
        end
        data_XY = csvread(name_files_XY{ifile});
        data_compass = csvread(name_files_pol{ifile});
        data_epg = csvread(name_files_epg{ifile});
        data_pen = csvread(name_files_pen{ifile});
        data_peg = csvread(name_files_peg{ifile});
        data_d7 = csvread(name_files_d7{ifile});
        data_no = csvread(name_files_no{ifile});
        data_pfn = csvread(name_files_pfn{ifile});
        data_pfnin = csvread(name_files_pinfn{ifile});
        data_hdcin = csvread(name_files_hindc{ifile});
        data_hdc = csvread(name_files_hdc{ifile});
        data_pfl = csvread(name_files_pfl{ifile});
        data_lal = csvread(name_files_lal{ifile});
        data_fbt2pfn_vecmemo = csvread(name_files_fbt_hd_vecmemo{ifile});
        data_fbt2hdc_vecmemo = csvread(name_files_fbt_pi_vecmemo{ifile});

        %% Behavioural data (X, Y and Oz data)
        X_uncorrected = data_XY(:, 1);
        Y_uncorrected = data_XY(:, 2);
        % Python-to-Matlab coordinate adjustement
        rot_rho = sqrt(X_uncorrected.^2 + Y_uncorrected.^2);
        rot_sig = atan2d(-Y_uncorrected, X_uncorrected);
        % Corrected data
        X = cosd(rot_sig) .* rot_rho;
        Y = sind(rot_sig) .* rot_rho;
        Oz = wrapTo180(-data_XY(:, 6));
        dOz = diff(Oz, 1);
        Oz_display = Oz;
        Oz_display(abs(dOz) > 180) = NaN;

        Target_position = [300*cosd(-45) 300*sind(-45)];
    
        figure(fig1_XYplot.mainfig)
        fig1_XYplot.tab(ifile) = uitab(fig1_XYplot.tabgroup);
        axes('Parent',fig1_XYplot.tab(ifile));

        subplot(3, 4, [1:2 5:6 9:10])
        hold on
        circle(Target_position(1), Target_position(2), 100, 0.1, 'color', 'b', 'linewidth', 0.1)
        circle(Target_position(1), Target_position(2), 15, 1.0, 'color', 'g', 'linewidth', 1)
        plot(X, Y,'k', 'LineWidth', 1.2)
        xlabel('X', 'FontSize', 11)
        ylabel('Y', 'FontSize', 11)
        daspect([1 1 1])

        subplot(3, 4, 3:4)
        plot(X, 'k', 'LineWidth', 1.2)
        hold on
        hline(Target_position(1), '', 'color', 'g')
        ylabel('Orientation (°)', 'FontSize', 9)
        % xlabel('Simulation steps', 'FontSize', 10)

        subplot(3, 4, 7:8)
        plot(Y, 'k', 'LineWidth', 1.2)
        hold on
        hline(Target_position(2), '', 'color', 'g')
        ylabel('Orientation (°)', 'FontSize', 9)
        % xlabel('Simulation steps', 'FontSize', 10)

        Target_relatori = atan2d(-(Y-Target_position(2)), X-Target_position(1));
        dT_relori = diff(Target_relatori, 1);
        Target_relatori_display = Target_relatori;
        Target_relatori_display(abs(dT_relori) > 180) = NaN;
        
        
        subplot(3, 4, 11:12)
        plot(Oz_display, 'k', 'LineWidth', 1.2)
        hold on
        plot(Target_relatori, 'g', 'Linewidth', 1.5)
        ylim([-180 180])
        ylabel('Orientation (°)', 'FontSize', 9)
        xlabel('Simulation steps', 'FontSize', 10)

        %% CX inputs
        figure(fig0_CXinputs.mainfig)
        fig0_CXinputs.tab(ifile) = uitab(fig0_CXinputs.tabgroup);
        axes('Parent',fig0_CXinputs.tab(ifile));
        
        subplot(2,1,1)
        imagesc(data_no')

        subplot(2,1,2)
        imagesc(data_compass')


        %% Compass circuit activity
        figure(fig2_compass.mainfig)
        fig2_compass.tab(ifile) = uitab(fig2_compass.tabgroup);
        axes('Parent',fig2_compass.tab(ifile));

        subplot(5,1,1);
        plot(Oz_display, 'r', 'LineWidth', 1.2)
        xlim([0 length(X)])
        ylim([-180 180])
        ylabel('Orientation (°)', 'FontSize', 10)

        subplot(5,1,2)
        imagesc(data_epg')
        ylabel('EPG #', 'FontSize', 11)

        subplot(5,1,3)
        imagesc(data_peg')
        ylabel('PEG #', 'FontSize', 11)

        subplot(5,1,4)
        imagesc(data_pen')
        ylabel('PEN #', 'FontSize', 11)

        subplot(5,1,5)
        imagesc(data_d7')
        ylabel('Δ7 #', 'FontSize', 11)
        xlabel('Simulation steps', 'FontSize', 15)

    end
end
waitbar(1,advance_bar,'All data extracted');

close(advance_bar)
cd(folder_simulation)

save_figure = 0;
if save_figure == 1
    if ~exist(fullfile(folder_simulation,'Figures'),'dir')
        mkdir(fullfile(folder_simulation,'Figures'))
    end
    folder_figures = fullfile(folder_simulation,'Figures');
end