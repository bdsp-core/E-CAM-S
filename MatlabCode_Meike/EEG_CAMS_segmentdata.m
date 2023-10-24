function segs = EEG_CAMS_segmentdata(data, window_size)
    % input: size(data) = (#channel, #points)
    % output: size(segment) = (#windows, #channel, #points in each windows)
    n_seg=floor(length(data')./window_size);
    for t = 1:n_seg
        seg_data(:,:)=data(:, (t-1)*window_size+1 : t*window_size);
        segs(t,:,:)=seg_data;
    end
end

    % start_ids contains the starting point of each window,
    % for example, window_size = 6*200 = 1200, and size(data)=(4,10000)
    % start_ids = [1, 1201, 2401, ...., 9601]
    
%     start_ids = linspace(1, size(resampleddata,2), window_size);
%     segs = [];
%     for st = start_ids
%         % take the st-th window, the start id of this window is
%         % (st-1)*window_size+1
%         % the end id of this window is
%         % st*window_size
%         % and then somehow concatenate to segs
%         segs = [segs data(1:end, (st-1)*window_size+1:st*window_size)];
%     end
%     
    