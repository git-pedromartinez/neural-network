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
  Matrix,
  NeuralNetworkConfig,
  NeuralNetworkData,
  TrainingNetworkData,
} from "../models";
import { NetworkStorageManager } from "../utils";

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

  constructor({
    sizes,
    learningRate,
    epochs,
    activationFunction = ActivationFunctions.sigmoid,
    activationDerivative = ActivationFunctions.sigmoidDerivative,
  }: NeuralNetworkConfig) {
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
      inputs.forEach((input, index) => {
        this.backward(input, targets[index]);
      });
    }
    this.log(`Training completed after ${this.epochs} epochs.`);
  }

  private backward(inputs: number[], targets: number[]): void {
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
        this.errorThreshold
      );
    }
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
    epsilon: number // Error threshold
  ): void {
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
      } else {
        // this.log(
        //   `The error ${errors[i]} is less than the epsilon threshold (${epsilon}), no weights or bias adjustments for neuron ${neuron.identifier}.`
        // );
      }
    });
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
    const TrainingNetworkData: TrainingNetworkData = {
      weights: this.layers.map((layer) =>
        layer.neurons.map((neuron) => [...neuron.weights])
      ),
      biases: this.layers.map((layer) =>
        layer.neurons.map((neuron) => neuron.bias)
      ),
    };
    // Save the training to a JSON file
    NetworkStorageManager.showLogs = this.showLogs;
    NetworkStorageManager.saveObject(TrainingNetworkData, this.trainingName);
    this.log("Training saved successfully.");
  }

  public loadTraining(): boolean {
    // Load the training from a JSON file
    NetworkStorageManager.showLogs = this.showLogs;
    const TrainingNetworkData: TrainingNetworkData | undefined =
      NetworkStorageManager.loadObject(this.trainingName);
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

    this.log(`Training loaded successfully.`);
    return true;
  }

  public log(...messages: any[]): void {
    if (this.showLogs) {
      console.log(...messages);
    }
  }
}
