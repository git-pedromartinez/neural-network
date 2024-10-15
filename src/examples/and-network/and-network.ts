/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { ActivationFunctions, NeuralNetwork, Neuron } from "../../core";
import { NeuralNetworkConfig } from "../../models";
import { testNetwork } from "../../utils";
import { AND_DATA } from "./data";

export function createNetwork() {
  Neuron.defaultWeights = 1;
  Neuron.defaultBias = 1;

  const network_config: NeuralNetworkConfig = {
    sizes: [2, 1],
    learningRate: 0.5,
    epochs: 10,
    activationFunction: ActivationFunctions.binaryStep,
    activationDerivative: ActivationFunctions.binaryStep,
  };
  const network: NeuralNetwork = new NeuralNetwork(network_config);
  network.saveHistory = true;
  network.trainingName = "and_training_optimized";
  network.errorThreshold = 0;
  // network.showLogs = true;

  //Training
  // network.train(AND_DATA);
  // network.saveTraining();

  //Testing
  network.loadTraining();
  testNetwork(network, AND_DATA);
}

createNetwork();
