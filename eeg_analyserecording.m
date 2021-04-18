%This script compares the fft of 2 different recordings
close all
clear variables
    %We expect to see greater alpha activity in the 'eyes closed' recording
fSampling = 250;
NFFT = 2048;
fVals=fSampling*(-NFFT/2:NFFT/2-1)/NFFT;
Px = zeros(1,NFFT);

cd recordings
files = dir('*.txt');
data = cell(length(files),1);
%dataVectorfilt = bandpass(data1,[2 40],250);
%dataVectorfilt = highpass(data1,2,250);
for i = 1:length(files)
    fid = fopen(files(i).name,'r');
    data{i} = fscanf(fid,'%d');
    fclose(fid);
    dataVectorfilt = data{i}-mean(data{i});
    X=fftshift(fft(dataVectorfilt,NFFT));         
    Px=X.*conj(X)/(NFFT*length(data{i}));
    figure(i)
    plot(fVals,Px,'b');
    title(files(i).name)
    xlim([0 60]);
    ylim([0 2])
end