#include <bits/stdc++.h>
#include "MultiLayerPerceptron.h"
#include "csvutil.h"
int main()
{
    vector<vector<double>> data;
    vector<double> label;
    vector<vector<double>> input_train;
    vector<double> label_train;
    vector<vector<double>> input_test;
    vector<double> label_test;

    cout<<"------Loading dataset------\n";
    ifstream dataset_file("./MNIST_dataset/train.csv");
    if (!dataset_file.is_open()) {
        cerr << "Could not open the dataset" << endl;
        return EXIT_FAILURE;
    }
    string line;
    int it = 0;
    int traindelim = 0.8*42000; // 80-20 split
    while (getline(dataset_file, line)){
        if(it==0)
        {
            //ignoring first line
            it++;
            continue;
        }
        vector<double> temp = parseCSVLine(line);
        vector<double> input;
        for(int i = 1; i < temp.size(); i++)
            input.push_back(temp[i]/256.0);
        if(it <= traindelim)
        {
            input_train.push_back(input);
            label_train.push_back(temp[0]);
        }
        else 
        {
            input_test.push_back(input);
            label_test.push_back(temp[0]);
        }
        it++;
    }
    dataset_file.close();

    /*cout<<"Train size:"<<input_train.size()<<" "<<input_train[0].size()<<"\n";
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < input_train[i].size(); j++)
            cout<<input_train[i][j]<<" ";
        cout<<"\nLabel:"<<label_train[i]<<"\n";
    }
    cout<<"Test size:"<<input_test.size()<<" "<<input_test[0].size()<<"\n";
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < input_test[i].size(); j++)
            cout<<input_test[i][j]<<" ";
        cout<<"\nLabel:"<<label_train[i]<<"\n";
    } */

    cout<<"------Initializing network------\n";
    MultiLayerPerceptron mlp = MultiLayerPerceptron({784, 200, 50, 10}, 1.0, 0.0001, false);
    //mlp.set_weights_gaussrand();
    
    cout<<"------Training started------\n";
    double MSE;
    double correct_train;
    double total_train;
    double correct_test;
    double total_test;
    int epochs = 1000;

    // training
    for (int i = 0; i < epochs; i++)
    {
        MSE = 0.0;
        correct_train = 0.0;
        total_train = 0.0;
        for(int j = 0; j < input_train.size(); j++)
        {
            MSE += mlp.bp(input_train[j], {label_train[j]});
            vector<double> op = mlp.run(input_train[j]);
            int lab = distance(op.begin(), max_element(op.begin(), op.end()));
            if(lab == (int)label_train[j])
                correct_train++;
            total_train++;
            if(j%100==0)
                cout<<"MSE: "<<MSE<<" Training accuracy: "<<100.0*(correct_train/total_train)<<"%\r";
        }
        MSE = MSE / (double)input_train.size();
        correct_test = 0.0;
        total_test = 0.0;
        for(int j = 0; j < input_test.size(); j++)
        {
            vector<double> op = mlp.run(input_test[j]);
            int lab = distance(op.begin(), max_element(op.begin(), op.end()));
            if(lab == (int)label_test[j])
                correct_test++;
            total_test++;
            if(j%100==0)
                cout<<"Testing accuracy: "<<100.0*(correct_test/total_test)<<"%\r";
        }
        cout<<"Epoch: "<<i+1<<" MSE: "<<MSE<<" Training accuracy: "<<100.0*(correct_train/total_train)<<"% Test accuracy: "<<100.0*(correct_test/total_test)<<"%\n";

    }
    return 0;
}