function [meantotpow] = EEG_CAMS_Multitaper(eeg_data, chan_names, Fs, ts, filename,i_file)

%% Multitaper PSD estimation
% eeg_data=selectdata;
% chan_names=datanames;

[num_ch, num_ts] = size(eeg_data);
nw=4; %60

%Multi-taper power spectral density estimation
seg_size = 6*Fs;
n_seg = floor(length(eeg_data')./seg_size);
totalpowdb_all = [];
for t = 1:n_seg
    % take the segment from the signal
    seg_data = eeg_data(:, (t-1)*seg_size+1 : t*seg_size);
    %saveas(seg_data,sprintf('seg_data_%s.mat',filename));
    % compute spectrum of this segment
    [pxx,f] = pmtm(seg_data',nw,[4:0.5:8],Fs); % pxx is power density (v2/hz)
    totalpower=sum(pxx, 1)*(f(2)-f(1));  % unit of totalpower is v2
    totalpowdb=pow2db(totalpower);
    
    % save the total power of this segment to totalpowdb_all
    totalpowdb_all = [totalpowdb_all; totalpowdb];
end
meantotpow = mean(totalpowdb_all, 2); %mean total power of all (in this case 4) derivations

%% Plot PSD for all derivations
% figure(2),
% title(filename,'Interpreter','none')
% t=tiledlayout(2,2);
% nexttile
%     plot(f,pxx(:,1))
%     xlabel('frequency(Hz)')
%     ylabel('Power density(v2/hz)')
%     xlim([0 30])   
%     title(chan_names{1},'Interpreter','none');
% nexttile
%     plot(f,pxx(:,2))
%     xlabel('frequency(Hz)')
%     ylabel('Power density(v2/hz)')
%     xlim([0 30])   
% title(chan_names{2},'Interpreter','none');
% nexttile
%     plot(f,pxx(:,3))
%     xlabel('frequency(Hz)')
%     ylabel('Power density(v2/hz)')
%     xlim([0 30])  
% title(chan_names{3},'Interpreter','none');
% nexttile
% plot(f,pxx(:,4))
%     xlabel('frequency(Hz)')
%     ylabel('Power density(v2/hz)')
%     xlim([0 30])
% title(chan_names{4},'Interpreter','none');
% filename=filename(1:end-4);
% saveas(figure(2),sprintf('PSD_%s.png',filename));

end