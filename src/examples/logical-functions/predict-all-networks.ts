/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { NeuralNetwork } from "../../core";
import { network_config } from "./network-config";
import { AND_DATA, OR_DATA, XOR_DATA } from "./data";
import { AND_TRAINING, OR_TRAINING, XOR_TRAINING } from "./constants";
import { testNetwork } from "../../utils";

let network: NeuralNetwork;

// network = new NeuralNetwork(network_config);
// network.trainingName = AND_TRAINING;
// network.loadTraining();
// testNetwork(network, AND_DATA);

// network = new NeuralNetwork(network_config);
// network.trainingName = OR_TRAINING;
// network.loadTraining();
// testNetwork(network, OR_DATA);

network = new NeuralNetwork(network_config);
network.trainingName = XOR_TRAINING;
network.showLogs = true;
network.loadTraining();
testNetwork(network, XOR_DATA);
