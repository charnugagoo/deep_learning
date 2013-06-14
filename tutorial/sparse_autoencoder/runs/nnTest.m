clear all;

testSize = 20/100;
dataPoints = 8000;
dataDim = 12;

x = round(rand(dataPoints, dataDim));
y = mod(sum(x, 2), 2);

x = tonndata(x, false, false);
y = tonndata(y, false, false);

xTest = x(1, end - round(length(x) * testSize) + 1:end);
yTest = y(1, end - round(length(x) * testSize) + 1:end);
xTrain = x(1, 1:end - round(length(x) * testSize));
yTrain = y(1, 1:end - round(length(x) * testSize));

accuracy = [];
l = 30;
for l = 1:1:20
    layers = [l];

    net = patternnet(layers);
    configure(net, xTrain, yTrain);
    net.trainParam.Time = 20;
    %net.divideParam.trainRatio = 1 - 2 * testSize;
    %net.divideParam.valRatio = testSize;
    %net.divideParam.testRatio = testSize;

    [net,tr] = train(net,xTrain,yTrain);
    

    yTestPred = net(xTest);
    yTestPred = cell2num(yTestPred);
    yTestMat = cell2num(yTest);

    yTestPred = round(yTestPred);
    tCorrect = sum(yTestPred == yTestMat);

    fprintf('Correctly classified: %f  Hidden nodes: %i \n', tCorrect/(testSize * dataPoints), l);
    accuracy = [accuracy tCorrect/(testSize * dataPoints)];
end

