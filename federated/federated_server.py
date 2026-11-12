import flwr as fl
import torch
import os


# =====================================================
# Metrics Aggregation
# =====================================================

def weighted_average(metrics):

    return {}


# =====================================================
# Custom Federated Strategy
# =====================================================

class SaveModelStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(
        self,
        server_round,
        results,
        failures
    ):

        print("\n" + "=" * 60)
        print(f"Federated Round {server_round}")
        print("=" * 60)

        print(f"Successful Clients : {len(results)}")
        print(f"Failed Clients     : {len(failures)}")

        aggregated_parameters, metrics = super().aggregate_fit(
            server_round,
            results,
            failures
        )

        if aggregated_parameters is not None:

            print("\nSaving Global Federated Model...")

            ndarrays = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            os.makedirs(
                "weights",
                exist_ok=True
            )

            torch.save(
                ndarrays,
                "weights/federated_sannet_model.pth"
            )

            print(
                "Global Model Saved Successfully"
            )

            print(
                "Location : weights/federated_sannet_model.pth"
            )

        return aggregated_parameters, metrics


# =====================================================
# Federated Strategy
# =====================================================

strategy = SaveModelStrategy(

    # Use every hospital
    fraction_fit=1.0,

    # Skip evaluation
    fraction_evaluate=0.0,

    # Wait for all hospitals
    min_fit_clients=5,

    min_available_clients=5,

    min_evaluate_clients=0,

    evaluate_metrics_aggregation_fn=weighted_average

)


# =====================================================
# Start Server
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Starting Federated Server...")
    print("=" * 60)

    print("Hospitals Required : 5")
    print("Federated Rounds   : 3")
    print("=" * 60)

    fl.server.start_server(

        server_address="0.0.0.0:8080",

        config=fl.server.ServerConfig(

            num_rounds=3

        ),

        strategy=strategy

    )