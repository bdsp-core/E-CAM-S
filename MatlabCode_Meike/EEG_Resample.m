function resampleddata = EEG_Resample(eeg_data, chan_names, Fs_orig)
%% Resample data?
% Sampling frequencies Fs in the initial data set includes 200, 256, 512, 600
% Need to resample to common Fs vs. extract frequency specific data
Fs = 200; % Target sampling frequency: The majority of the data are at 512
if Fs_orig ~= Fs
    % Sampling frequency does not equal target, resample
    % For debugging consider making a copy, but did not do so here to conserve memory
    resampleddata = resample(eeg_data', Fs, Fs_orig)'; % Expects data in columns, not rows
else resampleddata=eeg_data;
end
end