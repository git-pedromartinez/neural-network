/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { Layer } from "./layer";
import { ActivationFunctions } from "./activation-functions";
import {
  ActivationFunction,
  ActivationFunctionDerivative,
  InputData,
  Matrix,
  NetworkMetaData,
  NeuralNetworkConfig,
  NeuralNetworkData,
  TrainingHistory,
  TrainingNetworkData,
} from "../models";
import { StorageManager } from "../utils";

export const NetworkStore = new StorageManager<TrainingNetworkData>();
export const NetworkHistoryStore = new StorageManager<TrainingHistory>();

export class NeuralNetwork {
  private layers: Layer[];
  private learningRate: number;
  private epochs: number;

  /**
   * @variable `_trainingName` defines the name of the training, this name is used to save and load the training in the training history
   */
  private _trainingName: string = `NeuralNetworkTraining_${new Date().getTime()}`;
  public get trainingName(): string {
    return this._trainingName;
  }
  public set trainingName(value: string) {
    this._trainingName = value;
  }

  /**
   * @variable `_errorThreshold` defines the error threshold, this variable can be adjusted to improve the model's accuracy, default is 0.01
   */
  private _errorThreshold: number = 0.01;
  public get errorThreshold(): number {
    return this._errorThreshold;
  }
  public set errorThreshold(value: number) {
    this._errorThreshold = value;
  }

  /**
   * @variable `_showLogs` defines whether logs are shown in the console, default is false, and can be adjusted at runtime
   * @default false
   */
  private _showLogs: boolean = false;
  public get showLogs(): boolean {
    return this._showLogs;
  }
  public set showLogs(value: boolean) {
    this._showLogs = value;
  }

  private metaData: NetworkMetaData = {};
  /**
   * Experimental feature: simulator
   */
  private trainingHistory: TrainingHistory = {
    metaData: this.metaData,
    epochs: [],
  };

  constructor({
    sizes,
    learningRate,
    epochs,
    activationFunction = ActivationFunctions.sigmoid,
    activationDerivative = ActivationFunctions.sigmoidDerivative,
  }: NeuralNetworkConfig) {
    this.metaData = {
      sizes,
      learningRate,
      epochs,
      activationFunction,
      activationDerivative,
    };
    this.layers = [];
    for (let i = 1; i < sizes.length; i++) {
      this.layers.push(
        new Layer(
          sizes[i],
          sizes[i - 1],
          activationFunction,
          activationDerivative,
          `Layer[${i}]`
        )
      );
    }
    this.learningRate = learningRate;
    this.epochs = epochs;
  }

  train(TrainingNetworkData: NeuralNetworkData[]): void {
    const inputs: Matrix<"2D"> = TrainingNetworkData.map((d) => d.inputs);
    const targets: Matrix<"2D"> = TrainingNetworkData.map((d) => d.targets);

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      // Inicializa los datos de la época
      const epochData: InputData = [];

      inputs.forEach((input, index) => {
        const outputs = this.forward(input);
        const inputData = {
          input,
          data: {
            targents: targets[index],
            outputs: outputs[outputs.length - 1],
            errors: [],
            weights: this.layers.map((layer) =>
              layer.neurons.map((neuron) => [...neuron.weights])
            ),
            biases: this.layers.map((layer) =>
              layer.neurons.map((neuron) => neuron.bias)
            ),
            gradients: [],
          },
        };

        this.backward(input, targets[index], inputData.data);
        epochData.push(inputData);
      });

      this.trainingHistory.epochs.push(epochData);
    }
    this.log(`Training completed after ${this.epochs} epochs.`);
    this.saveTrainingHistory();
  }

  private backward(inputs: number[], targets: number[], inputData: any): void {
    const outputs: Matrix<"2D"> = this.forward(inputs);
    const outputErrors: number[] = targets.map(
      (t, i) => t - outputs[outputs.length - 1][i]
    );
    const errors: Matrix<"2D"> = this.calculateErrors(outputErrors);

    for (let i = this.layers.length - 1; i >= 0; i--) {
      this.adjustWeights(
        this.layers[i],
        outputs[i],
        outputs[i + 1],
        errors[i],
        this.errorThreshold,
        inputData
      );
    }

    inputData.errors.push(errors);
  }

  private forward(inputs: number[]): Matrix<"2D"> {
    const outputs: Matrix<"2D"> = [inputs];
    for (const layer of this.layers) {
      outputs.push(layer.forward(outputs[outputs.length - 1]));
    }
    return outputs;
  }

  private adjustWeights(
    layer: Layer,
    inputs: number[],
    outputs: number[],
    errors: number[],
    epsilon: number, // Error threshold
    epochData: any
  ): void {
    const layerGradients: number[][] = [];
    layer.neurons.forEach((neuron, i) => {
      // Check if the error is greater than epsilon
      if (Math.abs(errors[i]) > epsilon) {
        // Calculate the gradient only if the error is greater than epsilon
        const gradient: number =
          errors[i] *
          neuron.activationDerivative(outputs[i]) *
          this.learningRate;
        // Adjust the bias and weights only if the error exceeds epsilon
        neuron.bias += gradient;
        neuron.weights = neuron.weights.map(
          (weight, j) => weight + inputs[j] * gradient
        );
        layerGradients.push(neuron.weights);
      } else {
        this.log(
          `The error ${Math.abs(
            errors[i]
          )} is less than the epsilon threshold (${epsilon}), no weights or bias adjustments for neuron ${
            neuron.identifier
          }.`
        );
      }
    });
    epochData.gradients.push(layerGradients);

    // Actualiza los pesos y sesgos en el historial
    epochData.weights = this.layers.map((layer) =>
      layer.neurons.map((neuron) => [...neuron.weights])
    );
    epochData.biases = this.layers.map((layer) =>
      layer.neurons.map((neuron) => neuron.bias)
    );
  }

  private calculateErrors(outputErrors: number[]): Matrix<"2D"> {
    const errors: Matrix<"2D"> = [outputErrors];
    for (let i = this.layers.length - 1; i > 0; i--) {
      const layer: Layer = this.layers[i];
      const previousLayer: Layer = this.layers[i - 1];
      const hiddenErrors: number[] = previousLayer.neurons.map((neuron, j) =>
        errors[0].reduce(
          (sum, error, k) => sum + error * layer.neurons[k].weights[j],
          0
        )
      );
      errors.unshift(hiddenErrors);
    }
    return errors;
  }

  predict(inputs: number[]): number[] {
    return this.forward(inputs)[this.layers.length];
  }

  public saveTraining(): void {
    this.metaData = {
      ...this.metaData,
      errorThreshold: this.errorThreshold,
      trainingName: this.trainingName,
    };
    const TrainingNetworkData: TrainingNetworkData = {
      metaData: this.metaData, // Añadir esta línea
      weights: this.layers.map((layer) =>
        layer.neurons.map((neuron) => [...neuron.weights])
      ),
      biases: this.layers.map((layer) =>
        layer.neurons.map((neuron) => neuron.bias)
      ),
    };
    NetworkStore.showLogs = this.showLogs;
    NetworkStore.saveObject(TrainingNetworkData, this.trainingName);
    this.log("Training saved successfully.");
  }

  public loadTraining(): boolean {
    NetworkStore.showLogs = this.showLogs;
    const TrainingNetworkData: TrainingNetworkData | undefined =
      NetworkStore.loadObject(this.trainingName);
    if (!TrainingNetworkData) {
      this.log(`Training was not loaded.`);
      return false;
    }

    this.layers.forEach((layer, i) => {
      layer.neurons.forEach((neuron, j) => {
        neuron.weights = [...TrainingNetworkData.weights[i][j]];
        neuron.bias = TrainingNetworkData.biases[i][j];
      });
    });

    this.metaData = TrainingNetworkData.metaData; // Añadir esta línea
    this.log(`Training loaded successfully.`);
    return true;
  }

  /**
   * Experimental feature: simulator
   */
  private saveTrainingHistory(): void {
    this.trainingHistory.metaData = {
      ...this.metaData,
      errorThreshold: this.errorThreshold,
      trainingName: this.trainingName,
    };
    NetworkHistoryStore.showLogs = this.showLogs;
    NetworkHistoryStore.saveObject(
      this.trainingHistory,
      `${this.trainingName}_history`
    );
    this.log("Training history saved successfully.");
  }

  public log(...messages: any[]): void {
    if (this.showLogs) {
      console.log(...messages);
    }
  }
}
