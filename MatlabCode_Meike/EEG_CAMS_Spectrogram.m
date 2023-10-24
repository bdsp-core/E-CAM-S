function features = EEG_CAMS_Spectrogram(eeg_data, chan_names, Fs, ts, filename)

%% Check that data is valid
if isempty(eeg_data)
    features = []; % Make class to make sure have blank data where needed?
    return;
end

%% Spectrogram estimation
%eeg_data=selectdata;
%chan_names=datanames;

[num_ch, num_ts] = size(eeg_data);
num_sec = 4;
win_samples = Fs*num_sec;
frac_overlap = 1 - (1/num_sec);

clear p;

% h=figure(1),
title(filename,'Interpreter','none')
% t=tiledlayout(2,2);
% nexttile
spectrogram(eeg_data(1,:), win_samples, floor(win_samples*frac_overlap), [], Fs,'yaxis');
title(filename,'Interpreter','none')
%title(chan_names{1},'Interpreter','none');
caxis([-10 20])
ylim([0 30])
% nexttile
% spectrogram(eeg_data(2,:), win_samples, floor(win_samples*frac_overlap), [], Fs,'yaxis');
% title(chan_names{2},'Interpreter','none');
% caxis([-10 20])
% ylim([0 30])
% nexttile
% spectrogram(eeg_data(3,:), win_samples, floor(win_samples*frac_overlap), [], Fs,'yaxis');
% title(chan_names{3},'Interpreter','none');
% caxis([-10 20])
% ylim([0 30])
% nexttile
% spectrogram(eeg_data(4,:), win_samples, floor(win_samples*frac_overlap), [], Fs,'yaxis');
% title(chan_names{4},'Interpreter','none');
% caxis([-10 20])
% ylim([0 30])
% currentFigure=gcf;
% title(filename,'Interpreter','none')
filename=filename(1:end-4);
% fpath= 'C:\Users\Meike\Dropbox (Partners HealthCare)\EEG_CAMS\Delrium_NonICU_GretaMarinka\DataEEG\Spectrograms';
% saveas(h,sprintf('Spectrogram_uncropped_%s.png',filename));

end
%%

% 
% h=figure(1),
% spectrogram(meandata,win_samples, floor(win_samples*frac_overlap), [], Fs,'yaxis')
% ylim([0 30])
% title(filename)
% caxis([-30 20])
% filename=filename(1:end-4)
% saveas(h,sprintf('SpecBi_%s.png',filename));

%% Plot individual channel spectrograms
% clf;
%  for i_ch = 1:num_ch
%     PlotSpecGram(ts/60, f, SpecDb(p(:, :, i_ch)));
%     title(chan_names{i_ch});
% %     temp_name = chan_names(i_ch, :);
% %     if numel(temp_name) > 1
% %         title(sprintf('%s-%s', chan_names{i_ch, 1}, chan_names{i_ch, 2}));
% %     else
% %         title(StringFromVariousDataTypes(chan_names(i_ch, :)));
% %     end
%     if i_ch > num_row
%         ylabel('');
%         set(gca, 'YTickLabel', '');
%     end
%     if mod(i_ch, num_row)
%         xlabel('');
%         set(gca, 'XTickLabel', '');
%     else
%         xlabel('Time (min)');
%     end
% end
% for ch_end = i_ch+1:numel(grid_axes)
%     set(grid_axes(ch_end), 'Visible', 'off');
% end
% 
% % %% Adjust axes
% c_lim = [-10 20];
% f_lim = [0 40];
% % t_lim = [1e3 2e3];
% % t_lim = [1e2 8e2];
% % t_lim = [4e2 14e2];
% t_lim = [-inf inf];
% 
% AxesSharedCAxis(grid_axes, c_lim);
% AxesSharedLimits(grid_axes, [t_lim, f_lim]);
% TitleSuper(filename);
% ExportPNG(filename(1:end-4));


