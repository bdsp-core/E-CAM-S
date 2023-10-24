function [Sdata, stimes, sfreqs] = fcn_computeSpec_avg(data, params)
    [Sdata, stimes,sfreqs]=mtspecgram_jj(data(1, :)',params,params.movingwin,1); 
    for k =2:size(data, 1) 
         
        s = mtspecgram_jj(data(k,:)',params,params.movingwin,1); 
        Sdata = Sdata+s;
     
    end
    Sdata = Sdata'/size(data, 1);
end
        