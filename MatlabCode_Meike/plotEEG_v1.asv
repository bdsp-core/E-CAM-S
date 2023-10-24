%% Plot EEG script
clc
clear all
close all

%% Path settings for script
% Dependencies copied into shared dropbox folder 2019/12/23: add current path and set paths in relation to it
mfile_full = matlab.desktop.editor.getActiveFilename;
[dir_mfile, mfile] = fileparts(mfile_full);
addpath(dir_mfile);

cell_dir = split(dir_mfile, filesep);
dir_clin = fullfile(cell_dir{1:end-1}, filesep, 'DataClinical');
dir_spec = fullfile(cell_dir{1:end-1}, filesep, 'Spectrograms');
dir_code = fullfile(cell_dir{1:end-1}, filesep, 'CodeEyal');
dir_eeg = fullfile(cell_dir{1:end-1}, filesep, 'DataEEG');
dir_features = fullfile(cell_dir{1:end-1}, filesep, 'DataFeatures');

%% Load True + predicted CAMS scores
load('ytrue.mat')
load('ypred.mat')
%% Load in EEGs
freq_min=0.5;
freq_max=30;
flag_notch = true;

% Find files
cd(dir_eeg);
files = dir('DELIRIUM_NON_ICU_AMSD*.mat');

filenames={files.name};
[a,~, ~, ~] = regexp(filenames, 'DELIRIUM_NON_ICU_AMSD([0-9]{3})', 'tokens');
temp = [a{:}];
subj = str2double(vertcat(temp{:}));
subjectID=array2table(subj);

for i_file = 4:numel(files)
    filename = files(i_file).name;
    % Load data
    load(filename, 'Fs', 'channels', 'data');               % Fs, channels, data, start_time
    ts = (0:size(data, 2)-1)/Fs;                            % Convert from samples to timestamps based on Fs (s)
    channels = EEG_CAMS_ChannelsToCell(channels);           % Standardize channels format, some saved as char and some as nested cell
   
    % % Zero mean each signal, e.g. remove DC Offset
    med_data= median(data,2);
    zero_data = bsxfun(@minus, data, med_data);
    clear data med_data;
     
    % Filter data for EEG freqs
    filt_data = FilterEEG(zero_data', Fs, freq_min, freq_max, flag_notch)'; % Data expected in columns rather than rows
    clear zero_data;
    
    % Calculate bipolar leads/refs/montage: standardize extraction
    [bipolar_data, bipolar_names, bipolar_abbrev] = EEG_CAMS_Leads(filt_data, channels);
    clear filt_data;
    selectdata=bipolar_data([2,3,7,9],:); 
    datanames=bipolar_abbrev([2,3,7,9],:);
    clear bipolar_data;
     
    % Resample data when necessary
    resampleddata = EEG_Resample(selectdata, datanames, Fs);
    
    % Segment data into 15s windows   (plot length)
    % size(data) = (#channel, #points)
    % size(segment) = (#windows, #channel, #points in each window)
    Fs_resamp=200;
    window_size=15*Fs_resamp;
    segs = EEG_CAMS_segmentdata(resampleddata, window_size); 
    clear resampleddata
    
    % remove segments with artefacts
    artefact = EEG_decide_artifact(segs);
    segs_wa = segs(artefact==0,:,:); % only segments without artefacts selected
    
    % Plot random 15 second window
    % choose middle window of recording
    segnum=round(size(segs_wa,1)./2);
    % Select 1 window
    plotwindow=squeeze(segs_wa(segnum,:,:));
    
    data=plotwindow;
      
    f = figure('units','normalized','outerposition',[0 0 1 1]);
    
    hold on
        [M, N] = size(plotwindow);
        
        % grid vertical lines per second %    
        for ii = 1:(N/Fs_resamp)-1
            plot([ii ii], [.5, 4.5], '--', 'color', [.7 .7 .7])
        end
            
        tt = 1:N;
        zscale = 1/150;
        dcOff = fliplr(1:M);
        % EEG %
        for ii = 1:size(plotwindow, 1)
            plot(tt/Fs_resamp, plotwindow(ii,:)*zscale+dcOff(ii), 'k');
            
        end
        
        % Scores %
        y = ytes2(1,i_file);
        y_hat = yptes2(1,i_file);
        title(['Ground truth: ', num2str(y), '  Prediction: ', num2str(y_hat)])
    hold off
    set(gca, 'ytick', 1:M, 'yticklabels', flipud(strrep(datanames, '_', '-')), 'xlim', [0 N/Fs_resamp]);
    box on
    xlabel('Time [sec]')
   
    % Save plot 
    subjID=subj(i_file,1);
    file_save = sprintf('%s%cPlotEEG_subject%s', dir_features, filesep, num2str(subjID));
    save(file_save,'f');
    
end



