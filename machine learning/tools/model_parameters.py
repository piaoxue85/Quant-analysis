import json

class ModelParam:
    def __int__(self):
        self.setparam()

    def setparam(self, epochs = 5, batch_size=32, loss = 'mse', optimizer = 'adam',
                  layer1_type = 'lstm', layer1_neurons = 100, layer1_return_seq = True,
                  layer2_type = 'dropout', layer2_rate = 0.2,
                  layer3_type = 'lstm', layer3_neurons=100, layer3_return_seq = True,
                  layer4_type = 'lstm', layer4_neurons=100, layer4_return_seq = False,
                  layer5_type = 'dropout', layer5_rate = 0.2,
                  layer6_type = 'dense', layer6_neurons = 1, activation = 'linear'):

        training_param= {
                "epochs": epochs,
                "batch_size": batch_size
            }
        model_param = {
                "loss": loss,
                "optimizer": optimizer,
                "layers": [
                    {
                        "type": layer1_type,
                        "neurons": layer1_neurons,
                        "input_timesteps": 49,
                        "input_dim": 6,
                        "return_seq": layer1_return_seq
                    },
                    {
                        "type": layer2_type,
                        "rate": layer2_rate
                    },
                    {
                        "type": layer3_type,
                        "neurons": layer3_neurons,
                        "return_seq": layer3_return_seq
                    },
                    {
                        "type": layer4_type,
                        "neurons": layer4_neurons,
                        "return_seq": layer4_return_seq
                    },
                    {
                        "type": layer5_type,
                        "rate": layer5_rate
                    },
                    {
                        "type": layer6_type,
                        "neurons": layer6_neurons,
                        "activation": activation
                    }
                ]
            }

        with open('config.json') as file:
            config = json.load(file)
        config["training"] = training_param
        config["model"] = model_param
        with open('config.json', 'w') as file:
            json.dump(config, file)