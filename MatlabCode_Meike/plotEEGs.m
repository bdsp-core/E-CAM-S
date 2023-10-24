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

for i_file = 1:numel(files)
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
%   selectdata=bipolar_data([2,4,24,59],:);
%   datanames=bipolar_abbrev([2,4,24,59],:);
    %clear bipolar_data;
     
    % Resample data when necessary
    resampleddata = EEG_Resample(selectdata, datanames, Fs);
    
    % Segment data into 15s windows   (plot length)
    % size(data) = (#channel, #points)
    % size(segment) = (#windows, #channel, #points in each window)
    Fs_resamp=200;
    window_size=15*Fs_resamp;
    segs = EEG_CAMS_segmentdata(resampleddata, window_size); 
    %clear resampleddata
    
    % remove segments with artefacts
    artefact = EEG_decide_artifact(segs);
    segs_wa = segs(artefact==0,:,:); % only segments without artefacts selected
    
    %% Plot random 15 second window
    % choose middle window of recording
    segnum=size(segs_wa,1)./2;
    % Select 1 window
    plotwindow=squeeze(segs_wa(25,:,:));
    
    data=plotwindow;
%     Fs = 200;
%     file_save=('EEGdata', dir_features, filesep)
%     save(file_save);
      
    figure(1),
    plot((1:1:3000)/Fs,plotwindow(1,:)+100)
    hold on
    plot((1:1:3000)/Fs,plotwindow(2,:)+30)
    plot((1:1:3000)/Fs,plotwindow(3,:)-30)
    plot((1:1:3000)/Fs,plotwindow(4,:)-100)
    hold off
    xlabel('Time (s)')

    %% Save plot 
    
end



