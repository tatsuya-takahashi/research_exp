%% readfile
% file = 'E:\999_tools\straight\legacy_STRAIGHT\src\vaiueo2d.wav';
% file = 'E:\999_tools\asano.wav';
for i = 1:9
    file = strcat('E:\002_datasets\006_recola\2018_AVEC\recordings_audio\recordings_audio\dev_', num2str(i), '.wav');
    [x, fs] = audioread(file);
    disp(file);
    disp(datetime);
    [f0raw,vuv,auxouts,prmouts]=MulticueF0v14(x,fs);
    writematrix(f0raw, strcat(file, '.csv'));
    writematrix(vuv, strcat(file, '_vuv.csv'));
end

for i = 1:9
    file = strcat('E:\002_datasets\006_recola\2018_AVEC\recordings_audio\recordings_audio\test_', num2str(i), '.wav');
    [x, fs] = audioread(file);
    disp(file);
    disp(datetime);
    [f0raw,vuv,auxouts,prmouts]=MulticueF0v14(x,fs);
    writematrix(f0raw, strcat(file, '.csv'));
    writematrix(vuv, strcat(file, '_vuv.csv'));
end


%% train
for i = 1:9
    file = strcat('E:\002_datasets\006_recola\2018_AVEC\recordings_audio\recordings_audio\train_', num2str(i), '.wav');
    [x, fs] = audioread(file);
    disp(file);
    disp(datetime);
    [f0raw,vuv,auxouts,prmouts]=MulticueF0v14(x,fs);
    writematrix(f0raw, strcat(file, '.csv'));
    writematrix(vuv, strcat(file, '_vuv.csv'));
end

%% extract F0


% disp(vuv);
% whos f0raw;
% disp(datetime);
% ap = exstraightAPind(x,fs,f0raw);
% disp(datetime);


%% plot
plot(f0raw);






%% debug
% Timeseries = 1:length(f0raw);

%% writer



%% loop
%for i = 1:9
%    disp(strcat('E:\002_datasets\006_recola\2018_AVEC\recordings_audio\recordings_audio\dev_', num2str(i), '.wav'));
%end