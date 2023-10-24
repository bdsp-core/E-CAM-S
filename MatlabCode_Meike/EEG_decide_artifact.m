function artefact = EEG_decide_artifact(segs)
abssegs=abs(segs);
%abssegs=median(abssegs,2); %mean abs over channels
limabs=[0 500];
maskabs= abssegs > limabs(end);

stdsegs=std(segs,[],3);
limstd= 1;
maskstd= stdsegs < limstd;

for ui = 1:size(segs,1)
    check1=ismember(1,maskabs(ui,:,:));
    check2=ismember(1,maskstd(ui,:));
     if  check1 == true  || check2 == true
        artefact(ui,:) = 1;
     else
         artefact(ui,:) = 0;
     end
end

end

% % car = median(data, 1); %median channels
% % data_car = data-car;
% % data_range = mad(data_car, 1); % Computes mean absolute deviation based on medians to eliminate influence of single bad channel
% % lim_range = [1 500];
% % mask_range = lim_range(1) < data_range & data_range < lim_range(end);