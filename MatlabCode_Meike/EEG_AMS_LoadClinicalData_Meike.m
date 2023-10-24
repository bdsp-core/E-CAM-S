function tbl = EEG_AMS_LoadClinicalData_Meike(dir_data)

if nargin < 1 || isempty(dir_data)
%     dir_data = [cdDropbox '\MATLAB\EEG_AMS'];
    dir_data = 'D:\Dropbox (Partners HealthCare)\EEG_CAMS\deliriumDataSpreadsheet_PHI_v2';
end

%% Load data
cd(dir_data);
files = dir('Master Delirium Spread sheet2*.xls*');
[sort_val, sort_idx] = sort([files.datenum]);
filename = files(sort_idx(end)).name;
fprintf('File = %s\n', filename);
[~, ~, raw] = xlsread(filename);
% [~,~,time]=XLSREAD(filename,EncephReeval,'P15:HQ15')

%% Backup data
raw_backup = raw;
% raw = raw_backup;

%% Get Severity Scores
[camSeverityScore, CAM_score, cam3D] = extractExcelInfo_v2_EK2(raw);
% [camSeverityScore, CAM_score, cam3D] = parseSeverityScores(raw);
mask_cam3D = cam3D > 0;
cam3D_score = mask_cam3D(1, :) & mask_cam3D(2, :) & (mask_cam3D(3, :) | mask_cam3D(4, :));
cam3D_severity = sum(cam3D);

cam3D_sub = sum(mask_cam3D) >= 2 & ~cam3D_score;
% [cam3D_sub; sum(mask_cam3D); cam3D_score]
cam3D_strat = cam3D_sub + cam3D_score*2;

%% Limit raw to columns starting from "Row labels"
[row_label, col_label] = find(strcmp(raw, 'Row labels'));
raw = raw(:, col_label:end);

%% Only keep first label column and subject ID columns (changed this %)
tag = 'AMSD';
[row_subj, col_subj] = find(strncmp(raw, tag, numel(tag)));
raw = raw(:, [1 col_subj(:)']);

%% Restrict raw to rows with data in the label column (changed this %)
% mask_row = cellfun(@isstr, raw(:, 1));
% raw = raw(mask_row, :);

%% Convert Subject IDs to numbers (changed this %)
% row_id = strcmp(raw(:, 1), 'SUBJECT ID');
% for i_col = 2:size(raw, 2)
%     str_id = raw{row_id, i_col};
%     raw{row_id, i_col} = str2double(str_id(5:end));
% end

%% Change Gender code from text to num
row_gender = strcmp(raw(:, 1), 'Gender');
raw(row_gender, 2:end) = mat2cell(double([raw{row_gender, 2:end}] == 'F'), 1, ones(1, size(raw(1, 2:end), 2)));
% row_gender = strcmp(raw(:, 1), 'Gender');
% raw(row_gender, 2:end) = mat2cell(double([raw{row_gender, 2:end}] == 'M'), 1, ones(1, size(raw(1, 2:end), 2)));

%% Change date strings to Matlab datenums
row_date = find(strncmp(raw(:, 1), 'Date of', 7) | strncmp(raw(:, 1), 'Day of', 6));
for i_row = 1:numel(row_date)
    for i_col = 2:size(raw, 2)
        date_str = raw{row_date(i_row), i_col};
        temp_date = NaN;
        if ischar(date_str) && ~isnan(date_str(1))
            try
                temp_date = datenum(date_str);
            catch
                temp_date = NaN;
            end
        end
        raw{row_date(i_row), i_col} = temp_date;
    end
end

%% Add in severity scores
cam3D_1 = cam3D(1, :);
cam3D_2 = cam3D(2, :);
cam3D_3 = cam3D(3, :);
cam3D_4 = cam3D(4, :);

vars = {'CAM_score', 'camSeverityScore', 'cam3D_score', 'cam3D_severity', 'cam3D_1', 'cam3D_2', 'cam3D_3', 'cam3D_4', 'cam3D_sub', 'cam3D_strat'};

for i_var = 1:numel(vars)
    temp_data = eval(vars{i_var});
    raw{end+1, 1} = vars{i_var};
    raw(end, 2:numel(temp_data) + 1) = num2cell(temp_data);
    raw(end, numel(temp_data) + 2:end) = num2cell(NaN);
end


%% Exclude subjects based on reasons for exclusion: e.g. dementia, or unintentional focality
if ~sum(strcmpi(raw(:, 1), 'Include'))
    raw{end+1, 1} = 'Include';
    raw(end, 2:end) = num2cell(true(1, size(raw, 2)-1), [1, size(raw, 2)-1]);
end
row_include = find(strcmpi(raw(:, 1), 'Include'));

str_exclusion = {'Dementia', 'Incomplete Data', 'No EEG', 'EEG Artifact'};
row_reason_exclusion = find(strcmpi(raw(:, 1), 'Reason for Exclusion'));
for i_str = 1:numel(str_exclusion)
    mask_exclusion = strcmpi(raw(row_reason_exclusion, :), str_exclusion{i_str});
    if sum(mask_exclusion) > 1
        raw(row_include, mask_exclusion) = num2cell(false(1, sum(mask_exclusion)), [1, sum(mask_exclusion)]);
    elseif sum(mask_exclusion) == 1
        raw{row_include, mask_exclusion} = false;
    end
end

mask_include = [raw{row_include, 2:end}];
raw = raw(:, [true mask_include]);


%% Catch all non numbers (changed this %)
% data = raw(:, 2:end);
% mask = ~(cellfun(@isnumeric, data) | cellfun(@islogical, data));
% for i_row = 1:size(raw, 1)
%     if sum(mask(i_row, :))
%         fprintf('%d: %s: %d/%d not#: ', i_row, raw{i_row, 1}, sum(mask(i_row, :)), numel(mask(i_row, :)));
%         fprintf('%d ', raw{row_id, 1 + find(mask(i_row, :))});
%         fprintf('\n');
%     end
% end
% fprintf('\n');
% 
% % However, preserve following text fields
% field_names = {'AdmitTeam', 'Subspecialty'};
% for i_field = 1:numel(field_names)
%     mask_field = strcmp(raw(:, 1), field_names{i_field});
%     mask(mask_field, :) = false;
% end
% % sum(mask(:))
% 
% % Make all non numbers = NaN
% data(mask) = {NaN};
% raw(:, 2:end) = data;

%% Change duplicate label names (should do in original spreadsheet) in order to create a table --> (changed this %)
% tab_labels = tabulate(raw(:, 1));
% counts = [tab_labels{:, 2}];
% mult_labels = find(counts > 1);
% for i_mult = 1:numel(mult_labels)
%     temp_label = tab_labels{mult_labels(i_mult), 1};
%     idx_label = find(strcmp(raw(:, 1), temp_label));
%     fprintf('%s ', temp_label);
%     fprintf('%d ', idx_label);
%     fprintf('\n');
%     for i_label = 2:numel(idx_label)
% %         raw{idx_label(i_label), 1} = sprintf('%s%d', temp_label, i_label);
%     end
% end

%% Create a Matlab table: Start --> (changed this %)
% Make valid names?
varnames = {};
for i_row = 1:size(raw, 1)
    temp_name = raw{i_row, 1};
    temp_name = regexprep(temp_name, '[ -?\[\] ]', '');
   
        
    if (temp_name(1) >= '0') && (temp_name(1) <= '9')
        temp_name = ['x' temp_name];
    end
    if numel(temp_name) > 40
        temp_name = temp_name(1:30);
    end
    varnames{i_row} = temp_name;
end


tbl = cell2table(data', 'VariableNames', varnames);
tbl_backup = tbl;

%% Make some variables categorical --> I think don't need this so %
% field_names = {'AdmitTeam'};
% % field_names = {'AdmitTeam', 'Subspecialty'}; % Sub doesn't work due to blanks?
% for i_field = 1:numel(field_names)
%     tbl.(field_names{i_field}) = categorical(tbl.(field_names{i_field}));
% end

%% Restrict subjects?
% tbl = tbl_backup;
% % tbl = tbl(tbl{:, 'CAM_score'} == 1, :);
% % tbl = tbl(tbl{:, 'CAM_score'} == 0, :);

%% Add data --> same holds for this so %
% tbl.Death = tbl.GOSathospitaldischarge==1;
% 
% tbl.CPT = tbl.SAVEAHAARTcorrect010;
% tbl.CPT(tbl.CPT == -1) = 4;
% 
% tbl.CPT_FN = tbl.SAVEAHAARTFN04;
% tbl.CPT_FN(tbl.CPT_FN == -1) = 4;
% tbl.CPT_FP = tbl.SAVEAHAARTFP06;
% tbl.CPT_FP(tbl.CPT_FP == -1) = 0;
% 
% tbl.CPT_TN = 6 - tbl.SAVEAHAARTFP06;
% tbl.CPT_TN(tbl.SAVEAHAARTFP06 == -1) = 6;
% tbl.CPT_TP = 4 - tbl.SAVEAHAARTFN04;
% tbl.CPT_TP(tbl.SAVEAHAARTFN04 == -1) = 0;
% 
% % For patients that did not follow instructions, scored as -1 originally
% % tbl.CPT = tbl.SAVEAHAARTcorrect010081017206;
% % tbl.CPT(tbl.CPT == -1) = NaN;
% % 
% % tbl.CPT_FN = tbl.SAVEAHAARTFN04;
% % tbl.CPT_FN(tbl.CPT_FN == -1) = NaN;
% % tbl.CPT_FP = tbl.SAVEAHAARTFP06;
% % tbl.CPT_FP(tbl.CPT_FP == -1) = NaN;
% % 
% % tbl.CPT_TN = 6 - tbl.SAVEAHAARTFP06;
% % tbl.CPT_TN(tbl.SAVEAHAARTFP06 == -1) = NaN;
% % tbl.CPT_TP = 4 - tbl.SAVEAHAARTFN04;
% % tbl.CPT_TP(tbl.SAVEAHAARTFN04 == -1) = NaN;
% 
% tbl.CPT_Hit = tbl.CPT_TP/4;
% tbl.CPT_FA = tbl.CPT_FP/6;
% 
% tbl.DayHospEEG = tbl.Dateofevaluation - tbl.Dayofadmissiontohospital;

%% Admission info for patients --> also think don't need this
% % Otherwise scored as NaN
% field_names = {'ICU', 'ConsultNeuro', 'ConsultPsych'};
% idx_dx = 1+find(strcmp(tbl.Properties.VariableNames, 'AdmissionDxEyal')):find(strcmp(tbl.Properties.VariableNames, 'Other'))-1;
% field_names = [field_names tbl.Properties.VariableNames(idx_dx)];
% 
% for i_field = 1:numel(field_names)
%     field_data = tbl.(field_names{i_field});
%     mask_nan = isnan(field_data);
%     tbl.(field_names{i_field})(mask_nan) = false;
% end    
% 
% label = 'AdmitTeam';
% name_team = unique(tbl.(label));
% for i_team = 1:numel(name_team)
%     field_name = sprintf('Admit%s', name_team{i_team});
%     tbl.(field_name) = strcmp(tbl.AdmitTeam, name_team{i_team});
% end

%% Create a joint EEG slowing measure
% tbl{:, 'EEGslowingGenOrFocal'} = sum(tbl{:, {'Thetaslowing', 'Deltaslowing'}}, 2) > 0;
% tbl{:, 'EEGslowGen'} = sum(tbl{:, {'Generalizedthetaslowing', 'Generalizeddeltaslowing'}}, 2) > 0;
% tbl{:, 'EEGslowGenOrRhythmic'} = sum(tbl{:, {'Generalizedthetaslowing', 'Generalizeddeltaslowing', 'GRDA'}}, 2) > 0;
% tbl{:, 'EEGAbsentPDR8Hz'} = ~tbl{:, 'PosteriorDominantRhythm8Hz'};
% tbl{:, 'EEGslowGenOrGRDAOrAbsPDR'} = sum(tbl{:, {'Generalizedthetaslowing', 'Generalizeddeltaslowing', 'GRDA', 'EEGAbsentPDR8Hz'}}, 2) > 0;
% tbl{:, 'EEGslowGenOrGRDAAndAbsPDR'} = tbl{:, 'EEGslowGenOrRhythmic'}  & tbl{:, 'EEGAbsentPDR8Hz'};
% tbl{:, 'EEGslowGenDeltaAndThetaAndAbsPDR'} = sum(tbl{:, {'Generalizedthetaslowing', 'Generalizeddeltaslowing', 'EEGAbsentPDR8Hz'}}, 2) == 3;
% tbl{:, 'EEGany'} = sum(tbl{:, {'Thetaslowing', 'Deltaslowing', 'LPDLateralized', 'GPDGeneralized', 'Triphasicwaves', 'LRDA', 'GRDA', 'LowvoltageGeneralizedattentuation', 'Burstsuppression', 'Sporadicdischarges', 'EEGAbsentPDR8Hz'}}, 2) > 0;
% tbl{:, 'EEGanyPD'} = sum(tbl{:, {'Sporadicdischarges', 'LPDLateralized', 'GPDGeneralized'}}, 2) > 0;

%% Done creating/adjusting/cleaning table
% Can export data from here for other analyses
