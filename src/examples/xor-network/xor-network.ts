/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { NeuralNetwork } from "../../core";
import { testNetwork } from "../../utils";
import { XOR_DATA } from "./data";

let network = new NeuralNetwork({
  sizes: [2, 3, 1],
  learningRate: 0.15,
  epochs: 10000,
});
network.trainingName = "XOR_TRAINING";
network.errorThreshold = 0.001;

//Train the network
network.train(XOR_DATA);
network.saveTraining();

//Test the network
network.loadTraining();
testNetwork(network, XOR_DATA);
