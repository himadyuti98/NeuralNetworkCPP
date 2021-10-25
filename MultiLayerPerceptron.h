#include <bits/stdc++.h>

using namespace std;

double frand(){
	return (2.0*(double)rand() / RAND_MAX) - 1.0;
}

double gaussrand(){
	static double U1, U2, S;
	//static int phase = 0;
	double X;
	try
	{
		//if(phase == 0) {
			do {
				U1 = frand();
				U2 = frand();
				S = U1 * U1 + U2 * U2;
				if(S==0 || S>=1)
					continue;
				X = U1 * sqrt(-2.0 * log(S) / S);
			} while(X < -1 || X > 1);

		//} else X = U2 * sqrt(-2 * log(S) / S);
		//phase = 1 - phase;
		//cout<<U1<<" "<<U2<<" "<<S<<" "<<X<<"\n";
		return X;
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
	
	
}

class Perceptron {
	public: 
		vector<double> weights;
		double bias;
		
		Perceptron(int inputs, double bias = 1.0, bool gaussRand = false)
		{
			this->bias = bias;
			this->weights.resize(inputs+1);
			if(gaussRand)
				generate(this->weights.begin(),this->weights.end(),gaussrand);
			else
				generate(this->weights.begin(),this->weights.end(),frand);
		}

        double run(vector<double> x)
		{
			x.push_back(this->bias);
			double sum = inner_product(x.begin(),x.end(),this->weights.begin(),(double)0.0);
			return this->sigmoid(sum);
		}
		
		void set_weights(vector<double> w_init)
		{
			this->weights = w_init;
		}
		
		double sigmoid(double x)
		{
			return 1.0/(1.0 + exp(-x));
		}
};

class MultiLayerPerceptron {
	public: 
		vector<int> layers;
		double bias;
		double eta;
		vector<vector<Perceptron> > network;
		vector<vector<double> > values;
		vector<vector<double> > d;
		
		MultiLayerPerceptron(vector<int> layers, double bias=1.0, double eta = 0.5, bool gaussRand = false)
		{
			this->layers = layers;
			this->bias = bias;
			this->eta = eta;

			for (int i = 0; i < this->layers.size(); i++){
				this->values.push_back(vector<double>(this->layers[i],0.0));
				this->d.push_back(vector<double>(this->layers[i],0.0));
				this->network.push_back(vector<Perceptron>());
				if (i > 0)
				{
					for (int j = 0; j < this->layers[i]; j++)
						this->network[i].push_back(Perceptron(this->layers[i-1], this->bias, gaussRand));
				}
			}
		}
		
		void set_weights(vector<vector<vector<double> > > w_init)
		{
			for (int i = 0; i< w_init.size(); i++)
				for (int j = 0; j < w_init[i].size(); j++)
					this->network[i+1][j].set_weights(w_init[i][j]);
		}

		/*void set_weights_gaussrand()
		{
			for(int i = 1; i < this->network.size(); i++)
			{
				for(int j = 0; j < this->network[i].size(); j++)
					this->network[i][j].set_weights(gaussrand());
			}
		}*/
		
		void print_weights()
		{
			cout << endl;
			for (int i = 1; i < this->network.size(); i++){
				for (int j = 0; j < this->layers[i]; j++) {
					cout << "Layer " << i+1 << " Neuron " << j << ": ";
					for (auto &it: this->network[i][j].weights)
						cout << it <<"   ";
					cout << endl;
				}
			}
			cout << endl;
		}
		
		vector<double> run(vector<double> x)
		{
			this->values[0] = x;
			for (int i = 1; i < this->network.size(); i++)
				for (int j = 0; j < this->layers[i]; j++)
					this->values[i][j] = this->network[i][j].run(this->values[i-1]);
			return this->values.back();
		}
		
		double bp(vector<double> x, vector<double> y)
		{
			// Feed a sample to the network
			vector<double> outputs = this->run(x);
			// Calculate the mean squared error
			vector<double> error;
			double mean_squared_error = 0.0;
			for (int i = 0; i < y.size(); i++){
				error.push_back(y[i] - outputs[i]);
				mean_squared_error += error[i] * error[i];
			}
			mean_squared_error /= this->layers.back();
			// Calculate the output error terms
			for (int i = 0; i < outputs.size(); i++)
				this->d.back()[i] = outputs[i] * (1 - outputs[i]) * (error[i]);

			// Calculate the error term of each unit on each layer    
			for (int i = this->network.size()-2; i > 0; i--)
				for (int h = 0; h < this->network[i].size(); h++){
					double fwd_error = 0.0;
					for (int k = 0; k < this->layers[i+1]; k++)
						fwd_error += this->network[i+1][k].weights[h] * this->d[i+1][k];
					this->d[i][h] = this->values[i][h] * (1-this->values[i][h]) * fwd_error;
				}
			
			// Calculate the deltas and update the weights
			for (int i = 1; i < this->network.size(); i++)
				for (int j = 0; j < this->layers[i]; j++)
					for (int k = 0; k < this->layers[i-1]+1; k++){
						double delta;
						if (k==this->layers[i-1])
							delta = this->eta * this->d[i][j] * this->bias;
						else
							delta = this->eta * this->d[i][j] * this->values[i-1][k];
						this->network[i][j].weights[k] += delta;
					}
			return mean_squared_error;
		}
};
