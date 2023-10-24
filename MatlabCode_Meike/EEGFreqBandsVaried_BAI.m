function [freq_bands, ratio_bands] = EEGFreqBandsVaried_BAI()

% freq_bands.deltaSlooter4 = [0.99 4]; % Preprint BJA 2018
% freq_bands.deltaSlooter5 = [0.99 5]; % Preprint BJA 2018
% freq_bands.deltaSlooter6 = [0.99 6]; % Preprint BJA 2018
% freq_bands.delta = [0.2 3];
% freq_bands.deltamin = [0.5 2];
% freq_bands.theta = [3 7];
% freq_bands.alpha = [7 12];
% freq_bands.alphamin = [8 11];
% freq_bands.beta = [12 30];
% freq_bands.betamin = [11 20];

freq_bands.delta = [0.5 4];
freq_bands.theta = [4 8];
freq_bands.alpha = [8 12];
freq_bands.deltaVtheta = [freq_bands.delta; freq_bands.theta];
freq_bands.deltaValpha = [freq_bands.delta; freq_bands.alpha];
freq_bands.thetaValpha = [freq_bands.theta; freq_bands.alpha];

end

% ratio_bands.delta_to_alpha = [0.99 4; 8 13];
% ratio_bands.delta_to_alphalow = [0.99 4; 7 12];
% ratio_bands.delta_to_alphabetamid = [0.99 4; 8 20];
% ratio_bands.delta_to_alphabeta = [0.99 4; 8 30];
% ratio_bands.deltathetamid_to_alpha = [0.99 6; 8 13];
% ratio_bands.deltathetamid_to_alphabetamid = [0.99 6; 8 20];
% ratio_bands.deltathetamid_to_alphabeta = [0.99 6; 8 30];
% ratio_bands.deltatheta_to_alpha = [0.99 8; 8 13];
% ratio_bands.deltatheta_to_alphabetamid = [0.99 8; 8 20];
% ratio_bands.deltatheta_to_alphabeta = [0.99 8; 8 30];
