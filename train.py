import torch
from accelerate import Accelerator
from GPTQE import GPTQE
from model import GPTConfig
import numpy as np
from hamiltonian import gen_hamiltonian
import os
import pandas as pd
import torch.distributed as dist
import json
import joblib
import holoviews as hv
import hvplot.pandas
import matplotlib.pyplot as plt
import pennylane as qml
from ReplayBuffer import ReplayBuffer
from math import ceil
import random


def build_operator_pool(n_qubits, t_values=None):

    if t_values is None:
        # ks = [1, 2, 3, 4, 5]
        # t_values = [np.power(2, k / 160) for k in ks]
        # t_values = [0.1, 0.2, 0.4, 0.8]
        # t_values = [0.05 * (2**i) for i in range(4)]
        t_values = [np.pi, np.pi/2, np.pi/3, np.pi/4, np.pi/8]
        t_values += [-t for t in t_values]  # add negatives

    pool = []

    # Two-qubit interactions (Z_i Z_{i+1})
    for i in range(n_qubits - 1):
        for t in t_values:
            pool.append(qml.PauliRot(2 * t, 'ZZ', wires=[i, i + 1]))
            pool.append(qml.PauliRot(2 * t, 'XX', wires=[i, i+1]))
            pool.append(qml.PauliRot(2 * t, 'YY', wires=[i, i+1]))

    # Single-qubit terms (X_i)
    for i in range(n_qubits):
        for t in t_values:
            pool.append(qml.PauliRot(2 * t, 'X', wires=[i]))
            pool.append(qml.PauliRot(2 * t, 'Y', wires=i))
            pool.append(qml.PauliRot(2 * t, 'Z', wires=i))

    # for i in range(n_qubits - 2):
    #     for t in t_values:
    #         pool.append(qml.PauliRot(2 * t, 'ZZ', wires=[i, i + 2]))
    #         pool.append(qml.PauliRot(2 * t, 'XX', wires=[i, i+2]))
    #         pool.append(qml.PauliRot(2 * t, 'YY', wires=[i, i+2]))

    return pool


dev = qml.device("default.qubit")


@qml.qnode(dev)
def energy_circuit(seq, ham, init_state, num_qubits):
    qml.BasisState(init_state, wires=range(num_qubits))

    for op in seq:
        qml.Snapshot(measurement=qml.expval(ham))
        qml.apply(op)
    return qml.expval(ham)


energy_circuit = qml.snapshots(energy_circuit)


def get_subsequence_energies(seq, hamiltonian, init_state, num_qubits):
    # Collates the energies of each subsequence for a batch of sequences
    energies = []
    for pool in seq:
        es = energy_circuit(pool, hamiltonian, init_state, num_qubits)
        energies.append(
            [es[k].item() for k in list(range(1, len(pool))) + ["execution_results"]]
        )
    return np.array(energies)


def main():
    accelerator = Accelerator()
    dir = "online_heisenberg7"
    os.makedirs(f"{dir}/histo", exist_ok=True)

    ham_label = 1
    num_qubits = 4
    seq_gen = 10 # PER EPOCH (IF GEN ITER IS 10, 10X CIRCUITS ARE GENERATED)
    seq_len = 6
    gen_iter = 1
    temperature = .1
    n_epochs = 1000
    n_batches = 10
    beta = .3
    buff_size = 100

    data = joblib.load(f'./VQE-generated-dataset/data/ground_state/0{num_qubits}qubit/label{ham_label}.jb')
    grd_E = data["ground_energy"]

    ham = gen_hamiltonian(ham_label, num_qubits)
    init_state = [0] * num_qubits
    op_pool = np.array(build_operator_pool(num_qubits), dtype=object)
    op_pool_size = len(op_pool)
    replay_buffer = ReplayBuffer(buff_size, seq_len, 2, accelerator.device)

    model = GPTQE(GPTConfig(
        vocab_size=op_pool_size + 1,
        block_size=seq_len,
        dropout=0.2,
        bias=False
    )).to("cpu")

    opt = model.configure_optimizers(
        weight_decay=0.01, learning_rate=5e-5, betas=(0.9, 0.95), device_type="auto"
    )

    model, opt = accelerator.prepare(model, opt)

    eval_iter = 50
    current_mae = 10000
    best_min = 10000

    losses = []
    true_Es_t = []
    pred_Es_t = []

    true_Es_g = []

    for epoch in range(0, n_epochs+1):
        if epoch % gen_iter == 0:
            model.eval()
            # if epoch < 10:
            #     seq_r = 1
            # elif epoch < 700:
            #     seq_r = .8
            # else:
            #     seq_r = .6
            seq_r = .7+.3*np.cos(epoch/100)

            with torch.no_grad():
                tokens, _ = model.module.generate(
                    n_sequences=int(seq_gen * gen_iter * seq_r),
                    max_new_tokens=seq_len,
                    temperature=temperature,
                    device="cuda",
                    hard_mask_repeats=True
                )

            gen_inds = (tokens[:, 1:] - 1).cpu().numpy()
            gen_op_seq = op_pool[gen_inds]

            energies = torch.from_numpy(get_subsequence_energies(gen_op_seq, ham, init_state, num_qubits)).to(accelerator.device)

            replay_buffer.update(tokens, energies)
            t, e = replay_buffer.sample(ceil((1-seq_r)*seq_gen*gen_iter))
            if t is not None:
                # for token_seq in t:
                #     for i in range(len(token_seq)):
                #         if token_seq[i] != 1 and random.random() < .3:
                #             token_seq[i] = random.randint(1, op_pool_size)

                # gen_inds = (t[:, 1:] - 1).cpu().numpy()
                # gen_op_seq = op_pool[gen_inds]
                # e = torch.from_numpy(get_subsequence_energies(gen_op_seq, ham, init_state, num_qubits)).to(accelerator.device)
                tokens = torch.cat((tokens, t), dim=0)
                energies = torch.cat((energies, e), dim=0)

            true_Es_g.append(energies[:, -1].cpu().numpy().reshape(-1, 1))
            train_inds = np.arange(len(tokens))

        model.train()
        np.random.shuffle(train_inds)
        token_batches = torch.tensor_split(tokens[train_inds], n_batches)
        energy_batches = torch.tensor_split(energies[train_inds], n_batches)
        loss_record = 0

        for token_batch, energy_batch in zip(token_batches, energy_batches):
            opt.zero_grad()
            loss = model.module.calculate_loss(token_batch, energy_batch, beta)
            accelerator.backward(loss)
            opt.step()
            loss_record += accelerator.gather_for_metrics(loss.detach()).mean().item()

        avg_loss = loss_record / n_batches
        losses.append(avg_loss)

        accelerator.print(f"Epoch {epoch}: Loss: {avg_loss:.4f}")

        if epoch != 0 and epoch % eval_iter == 0 and accelerator.is_main_process:
            # for gpt evaluation
            model.eval()
            with torch.no_grad():
                gen_token_seq, pred_Es = model.module.generate(
                    n_sequences=100,
                    max_new_tokens=seq_len,
                    temperature=temperature, # Use a low temperature to emphasize the difference in logits (play with temp)
                    device="cuda",
                    hard_mask_repeats=True
                )

            pred_Es = pred_Es.detach().cpu().numpy()

            gen_inds = (gen_token_seq[:, 1:] - 1).cpu().numpy()
            gen_op_seq = op_pool[gen_inds]

            true_Es = get_subsequence_energies(gen_op_seq, ham, init_state, num_qubits)[:, -1].reshape(-1, 1)

            mae = np.mean(np.abs(pred_Es - true_Es))
            ave_E = np.mean(true_Es)
            min_E = np.min(true_Es)

            pred_Es_t.append(pred_Es)
            true_Es_t.append(true_Es)

            print(f"Iteration: {epoch}, Loss: {losses[-1]}, MAE: {mae}, Ave E: {ave_E}, Min E: {min_E}")

            plt.figure(figsize=(10, 5))
            plt.hist(pred_Es, bins=30, alpha=0.6, label='Predicted Energy')
            plt.hist(true_Es, bins=30, alpha=0.6, label='Measured Energy')
            plt.axvline(min(true_Es), color='red', linestyle='--', label='Min Measured E')
            plt.axvline(sum(pred_Es)/len(pred_Es), color='black', linestyle='--', label='Average Predicted E')
            plt.legend()
            plt.title(f"Energy Distribution @ Epoch {epoch}")
            plt.xlabel("Energy")
            plt.ylabel("Count")
            plt.savefig(f"{dir}/histo/{epoch}")
            plt.close()

            if mae < current_mae:
                current_mae = mae
                min_epoch = epoch
                # save_dict = {
                #     "model_state_dict": model.state_dict() if hasattr(model, "module") else model.state_dict(),
                #     "optimizer_state_dict": opt.state_dict(),
                #     "epoch": epoch,  # optional
                # }
                # torch.save(save_dict, f"{dir}/checkpoint.pt")
                print(f"Model Saved at {epoch}")

            if min_E < best_min:
                best_min = min_E

    if accelerator.is_main_process:
        df_loss = pd.DataFrame(losses)
        df_loss.to_csv(f"{dir}/losses.csv", index=False)

        hvplot.extension('matplotlib')

        loss_fig = df_loss.hvplot(
            title="Training loss progress", ylabel="loss", xlabel="Training epochs", logy=True
        ).opts(fig_size=600, fontscale=2, aspect=1.2)

        hv.save(loss_fig, f"{dir}/loss_fig.png")

        pred_Es_t = np.concatenate(pred_Es_t, axis=1)
        true_Es_t = np.concatenate(true_Es_t, axis=1)

        df_pred = pd.DataFrame(pred_Es_t, columns=list(range(eval_iter, n_epochs+1, eval_iter)))
        df_true = pd.DataFrame(true_Es_t, columns=list(range(eval_iter, n_epochs+1, eval_iter)))

        df_pred.to_csv(f"{dir}/pred_Es_t.csv", index=False)
        df_true.to_csv(f"{dir}/true_Es_t.csv", index=False)

        df_true.columns = df_true.columns.astype(int)
        df_pred.columns = df_pred.columns.astype(int)

        df_trues_stats = pd.concat([df_true.mean(axis=0), df_true.min(axis=0), df_true.max(axis=0)], axis=1).reset_index()
        df_trues_stats.columns = ["Training Iterations", "Ave True E", "Min True E", "Max True E"]

        df_preds_stats = pd.concat([df_pred.mean(axis=0), df_pred.min(axis=0), df_pred.max(axis=0)], axis=1).reset_index()
        df_preds_stats.columns = ["Training Iterations", "Ave Pred E", "Min Pred E", "Max Pred E"]

        fig = (
            df_trues_stats.hvplot.scatter(x="Training Iterations", y="Ave True E", label="Mean True Energies") *
            df_trues_stats.hvplot.line(x="Training Iterations", y="Ave True E", alpha=0.5, linewidth=1) *
            df_trues_stats.hvplot.area(x="Training Iterations", y="Min True E", y2="Max True E", alpha=0.1)
        ) * (
            df_preds_stats.hvplot.scatter(x="Training Iterations", y="Ave Pred E", label="Mean Predicted Energies") *
            df_preds_stats.hvplot.line(x="Training Iterations", y="Ave Pred E", alpha=0.5, linewidth=1) *
            df_preds_stats.hvplot.area(x="Training Iterations", y="Min Pred E", y2="Max Pred E", alpha=0.1)
        )
        fig = fig * hv.Curve([[0, grd_E], [n_epochs+1, grd_E]], label="Ground State Energy").opts(color="k", alpha=0.4, linestyle="dashed")
        fig = fig.opts(ylabel="Sequence Energies", title="GQE Evaluations", fig_size=600, fontscale=2)
        hv.save(fig, f"{dir}/eval_fig.png")

        true_Es_g = np.concatenate(true_Es_g, axis=1)
        df_gen = pd.DataFrame(true_Es_g)
        df_gen.columns = df_gen.columns.astype(int)
        df_gen_stats = pd.concat([df_gen.mean(axis=0), df_gen.min(axis=0), df_gen.max(axis=0)], axis=1).reset_index()
        df_gen_stats.columns = ["Training Iterations", "Ave True E", "Min True E", "Max True E"]
        gen_fig = (
            df_gen_stats.hvplot.scatter(x="Training Iterations", y="Ave True E", label="Mean True Energies") *
            df_gen_stats.hvplot.line(x="Training Iterations", y="Ave True E", alpha=0.5, linewidth=1) *
            df_gen_stats.hvplot.area(x="Training Iterations", y="Min True E", y2="Max True E", alpha=0.1)
        )
        gen_fig = gen_fig.opts(ylabel="Sequence Energies", title="Generated Training Data", fig_size=600, fontscale=2)
        hv.save(gen_fig, f"{dir}/gen_fig.png")

        # save_dict = {
        #     "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.module.state_dict(),
        #     "optimizer_state_dict": opt.state_dict(),
        #     "epoch": epoch,  # optional
        # }
        # torch.save(save_dict, f"{dir}/final.pt")

        with open(f"{dir}/config.json", "w") as f:
            json.dump(model.module.config.__dict__, f, indent=4)

        metadata = {
            "num_qubits": num_qubits,
            "ham_label": ham_label,
            "epochs": n_epochs,
            "seq len": seq_len,
            "seq gen": seq_gen,
            "gen iter": gen_iter,
            "n batches": n_batches,
            "weight beta": beta,
            "buffer size": buff_size,
            "lowest energy": best_min,
            "final_loss": float(loss.item()),
            "min_mae": float(current_mae),
            "min_mae_epoch": min_epoch,
            "temperature": temperature,
        }

        with open(f"{dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()