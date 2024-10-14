/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { NeuralNetwork } from "../../core";
import { network_config } from "./network-config";
import { AND_DATA, OR_DATA, XOR_DATA } from "./data";
import {
  AND_TRAINING,
  errorThreshold,
  OR_TRAINING,
  XOR_TRAINING,
} from "./constants";

let network: NeuralNetwork;

network = new NeuralNetwork(network_config);
network.trainingName = AND_TRAINING;
network.errorThreshold = errorThreshold;
network.train(AND_DATA);
network.saveTraining();

network = new NeuralNetwork(network_config);
network.trainingName = OR_TRAINING;
network.errorThreshold = errorThreshold;
network.train(OR_DATA);
network.saveTraining();

network = new NeuralNetwork({ ...network_config, learningRate: 0.15 });
network.trainingName = XOR_TRAINING;
network.errorThreshold = errorThreshold;
network.train(XOR_DATA);
network.saveTraining();
