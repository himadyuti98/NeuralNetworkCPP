#include <iostream>
#include "MultiLayerPerceptron.h"

int main()
{
    MultiLayerPerceptron mlp = MultiLayerPerceptron({2,2,1});
    cout<<"\n\n--------XOR Example----------------\n\n";
    mlp = MultiLayerPerceptron({2,2,1}, 1.0, 0.5, false);
    cout<<"Training Neural Network as an XOR Gate...\n";
    double MSE;
    for (int i = 0; i < 3000; i++){
        MSE = 0.0;
        MSE += mlp.bp({0,0},{0});
        MSE += mlp.bp({0,1},{1});
        MSE += mlp.bp({1,0},{1});
        MSE += mlp.bp({1,1},{0});
        MSE = MSE / 4.0;
        if (i % 100 == 0)
            cout<<"MSE = "<<MSE<<endl;
    }

    cout<<"\n\nWeights:\n";
    mlp.print_weights();

    cout<<"\nXOR:"<<endl;
    cout<<"0 0 = "<<mlp.run({0,0})[0]<<endl;
    cout<<"0 1 = "<<mlp.run({0,1})[0]<<endl;
    cout<<"1 0 = "<<mlp.run({1,0})[0]<<endl;
    cout<<"1 1 = "<<mlp.run({1,1})[0]<<endl;

}