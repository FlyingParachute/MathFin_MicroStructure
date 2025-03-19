# Coursework of Market Microstructure

*MSc Mathematics and Finance, 2024-2025, Imperial College London*

This repository contains our implementation of queue-reactive models for limit order book (LOB) simulation as described in ["A Novel Approach to Queue-Reactive Models: The Importance of Order Sizes"](https://arxiv.org/abs/2405.18594) by Hamza Bodor and Laurent Carlier.

**Group Members:** Sunqinli Wang, Jinyi Lin, Xingjian Zhao

## Models Overview

Our implementation focuses on several variants of LOB models, each with different approaches to handling order sizes and arrival intensities:

### Baseline Model

The baseline model is used because of the limitation of high-frequency data of Bund future. And the simulation results of the baseline model act as "true" data to compare with the simulation results of other models.

### Queue-Reactive Model (QR) 

The QR model simulates order book dynamics by modeling order arrivals and cancellations as intensity-driven stochastic processes that react to the current queue size.

- Different types of order events: limit, market, and cancel orders
- Uses Poisson processes to capture order arrival patterns

### Size-Aware Queue-Reactive Model (SAQR)

This model considers the size of orders as an additional variable in determining the nature of order arrival.

- Conditional intensity matrix for order types and sizes
- Order sizes reflect the relationship between queue size and order size
- Enhanced modeling of queue consumption dynamics
- Large orders have different probabilities of execution and cancellation
- Specialized handling for small queue conditions

### Five-Type Queue-Reactive Model (FTQR)

FTQR extends the basic QR model by adding two additional event types:

- `cancelall`: Models complete withdrawal from a queue
- `marketall`: Models complete consumption of a queue by market orders

### Hawkes-Based Models

The Hawkes models incorporate:
- Base intensities for each event type
- Excitation matrices to capture how events impact future event probabilities
- Exponential decay for the excitation effect
- Cross-excitation between different order types

## Model Evaluation Methods

Our implementation adopts several metrics to evaluate the performance of the models:

- Mid-price dynamics
- Rolling 10-minute volatility
- Queue size distribution

## Data
In this project, various parameters of different models can be adjusted. We experimented with many configurations, and the notebook presents only a subset of them. Some representative historical simulation results can be found in the following link:
https://imperiallondon-my.sharepoint.com/:u:/g/personal/sw3121_ic_ac_uk/EQl4shBFR09JlDmY5PYO4T4B_PfFDueBeA6msNyKQThj-A?e=nal3IR
