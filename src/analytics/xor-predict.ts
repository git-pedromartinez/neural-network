/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { NeuralNetwork } from "../core";
import { testNetwork } from "../utils";
import { network_config, XOR_DATA, XOR_TRAINING } from "./network-config";

const network = new NeuralNetwork(network_config);
network.trainingName = XOR_TRAINING;
network.loadTraining();
testNetwork(network, XOR_DATA);
