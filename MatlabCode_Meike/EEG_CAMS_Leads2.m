function [bipolar_data, bipolar_names, bipolar_abbrev, unipolar_data, unipolar_names, ref_data] = EEG_CAMS_Leads2(data, channels)

%% Code
% % Selected leads
% lead_names = {
%     'Fp1', 'Fpz' % Forehead like: Noiseiest
%     'Fp2', 'Fpz' % Forehead like: Noiseiest
%     'Fp1', 'F7' % Forehead like
%     'Fp2', 'F8' % Forehead like
%     'Fp1', 'Fp2' % Forehead like
%     'Fp2', 'Pz' % Slooter: BJA 2018 Preprint
%     'T4', 'Pz' % Slooter: BJA 2018 Preprint: T8 = T4 based on MCN system
%     'Fp2', 'T4' % Slooter: BJA 2018 Preprint: T8 = T4 based on MCN system
%     
%     'F7', 'F8' % Horizontal: broad
%     'F7', 'Pz' % Slooter like
%     'F8', 'Pz' % Slooter like
%     'F7', 'P4' % Crossed parasagittal: lateral
%     'F8', 'P3' % Crossed parasagittal: lateral
%     'F7', 'O1' % More ant/post
%     'F8', 'O2' % More ant/post
%     
%     'F3', 'C3' % Unilateral parasagittal: small steps Ant
%     'F4', 'C4' % Unilateral parasagittal: small steps Ant
%     'C3', 'P3' % Unilateral parasagittal: small steps Post
%     'C4', 'P4' % Unilateral parasagittal: small steps Post
%     'F3', 'P3' % Unilateral parasagittal
%     'F4', 'P4' % Unilateral parasagittal
%     'F3', 'P4' % Unilateral parasagittal: Crossed
%     'F4', 'P3' % Unilateral parasagittal: Crossed
%     'F3', 'F4' % Cross anterior/medial
%     'P3', 'P4' % Cross posterior/medial
%     'F3', 'Pz' % Unilateral parasagittal: to midline
%     'F4', 'Pz' % Unilateral parasagittal: to midline
%     'F3', 'O1' % Unilateral parasagittal: to posterior
%     'F4', 'O2' % Unilateral parasagittal: to posterior
%     'Fz', 'P3' % Central out
%     'Fz', 'P4' % Central out
%     'Fz', 'Pz' % Central midline
%     'Fz', 'Cz' % Central midline
%     'Cz', 'Pz' % Central midline
%     'C3', 'C4' % Horizontal
%     'T3', 'T4' % Horizontal
%     'T3', 'Pz' % Horizontal
%     'T3', 'Pz' % Horizontal
%     'T5', 'O1' % Double Banana PDR
%     'T6', 'O1' % Double Banana PDR
%     'P3', 'O1' % Double Banana PDR
%     'P4', 'O2' % Double Banana PDR
%     };

% All leads
unipolar_names = {'Fp1', 'Fp2', 'Fpz', 'F7', 'F8', 'F3', 'F4', 'Fz', 'C3', 'C4', 'Cz', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2'};
bipolar_names = {};
for i_lead = 1:numel(unipolar_names)
    bipolar_names{end+1, 1} = unipolar_names{i_lead};
    bipolar_names{end, 2} = 'Na';
    for j_lead = i_lead+1:numel(unipolar_names)
        bipolar_names{end+1, 1} = unipolar_names{i_lead};
        bipolar_names{end, 2} = unipolar_names{j_lead};
    end
end
    
%% Make data for each lead
num_leads = size(bipolar_names, 1);
num_pts = size(data, 2);
bipolar_data = zeros(num_leads, num_pts);
bipolar_abbrev = cell(num_leads, 1);
for i_lead = 1:num_leads
    mask_lead_p = strcmp(channels, bipolar_names{i_lead, 1});
    mask_lead_n = strcmp(channels, bipolar_names{i_lead, 2});
    if num_pts
        if sum(mask_lead_p) && sum(mask_lead_n)
            bipolar_data(i_lead, :) = data(mask_lead_p, :) - data(mask_lead_n, :);
        elseif strcmp(bipolar_names{i_lead, 2}, 'Na')
            % Self lead
           try
            bipolar_data(i_lead, :) = data(mask_lead_p, :); 
           catch k
           end
        end
    end
    bipolar_abbrev{i_lead} = sprintf('%s_%s', bipolar_names{i_lead, 1}, bipolar_names{i_lead, 2}); % Use underscore instead of hyphen so that this can be used as a field name
end

% %% Plot some data
% clf;
% plot(lead_data');
% z = range(lead_data, 2);
% plot(z, '.');
% axis([0.5 num_leads+0.5 0 inf]);
