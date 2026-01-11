import numpy as np
import torch

import ai.graph
import ai.loss
import ai.parameter


def test_mse_loss_forward_backward():
    graph = ai.graph.ComputationalGraph()
    y_out_data = np.array([[1.0, -2.0], [3.0, 0.5]], dtype=float)
    y_true_data = np.array([[0.5, -1.5], [2.0, 1.0]], dtype=float)
    y_out = ai.parameter.Parameter(data=np.array(y_out_data, dtype=float), graph=graph)

    loss_fn = ai.loss.MSELoss(graph=graph)
    loss = loss_fn.forward(y_out, y_true_data)

    y_out_t = torch.tensor(np.array(y_out_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_true_t = torch.tensor(np.array(y_true_data, dtype=np.float64), dtype=torch.float64)
    loss_t = torch.nn.MSELoss(reduction="mean")(y_out_t, y_true_t)

    np.testing.assert_allclose(loss.data.squeeze(), loss_t.detach().numpy(), rtol=1e-6, atol=1e-6)

    loss.backward()
    loss_t.backward()

    np.testing.assert_allclose(y_out.grad, y_out_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_cross_entropy_loss_forward_backward():
    graph = ai.graph.ComputationalGraph()
    logits_data = np.array([[2.0, -1.0, 0.5], [0.0, 1.5, -0.5]], dtype=float)
    y_true_data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    logits = ai.parameter.Parameter(data=np.array(logits_data, dtype=float), graph=graph)

    loss_fn = ai.loss.CrossEntropyLoss(graph=graph)
    loss = loss_fn.forward(logits, y_true_data)

    logits_t = torch.tensor(np.array(logits_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_true_t = torch.tensor(np.array(y_true_data, dtype=np.float64), dtype=torch.float64)
    prob_t = torch.softmax(logits_t, dim=-1)
    loss_t = torch.nn.CrossEntropyLoss(reduction="mean")(torch.log(prob_t), torch.argmax(y_true_t, dim=-1))

    np.testing.assert_allclose(loss.data.squeeze(), loss_t.detach().numpy(), rtol=1e-6, atol=1e-6)

    loss.backward()
    loss_t.backward()

    np.testing.assert_allclose(logits.grad, logits_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_bce_loss_forward_backward():
    graph = ai.graph.ComputationalGraph()
    y_out_data = np.array([[0.2, 0.8], [0.7, 0.1]], dtype=float)
    y_true_data = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    y_out = ai.parameter.Parameter(data=np.array(y_out_data, dtype=float), graph=graph)

    loss_fn = ai.loss.BCELoss(graph=graph)
    loss = loss_fn.forward(y_out, y_true_data)

    y_out_t = torch.tensor(np.array(y_out_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_true_t = torch.tensor(np.array(y_true_data, dtype=np.float64), dtype=torch.float64)
    y_out_clamped = torch.clamp(y_out_t, 1e-8, 1.0 - 1e-8)
    loss_t = torch.nn.BCELoss(reduction="sum")(y_out_clamped, y_true_t) / y_true_t.shape[0]

    np.testing.assert_allclose(loss.data.squeeze(), loss_t.detach().numpy(), rtol=1e-6, atol=1e-6)

    loss.backward()
    loss_t.backward()

    np.testing.assert_allclose(y_out.grad, y_out_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_kl_div_loss_forward_backward():
    graph = ai.graph.ComputationalGraph()
    y_true_data = np.array([[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]], dtype=float)
    y_out_data = np.array([[0.3, 0.4, 0.3], [0.2, 0.3, 0.5]], dtype=float)
    y_out = ai.parameter.Parameter(data=np.array(np.log(y_out_data), dtype=float), graph=graph)

    loss_fn = ai.loss.KLDivLoss(graph=graph)
    loss = loss_fn.forward(y_out, y_true_data)

    y_out_t = torch.tensor(np.array(np.log(y_out_data), dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_true_t = torch.tensor(np.array(y_true_data, dtype=np.float64), dtype=torch.float64)
    loss_t = torch.nn.KLDivLoss(reduction="batchmean")(y_out_t, y_true_t)

    np.testing.assert_allclose(loss.data.squeeze(), loss_t.detach().numpy(), rtol=1e-6, atol=1e-6)

    loss.backward()
    loss_t.backward()

    np.testing.assert_allclose(y_out.grad, y_out_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_js_div_loss_forward_backward():
    graph = ai.graph.ComputationalGraph()
    y_true_data = np.array([[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]], dtype=float)
    y_out_data = np.array([[0.3, 0.4, 0.3], [0.2, 0.3, 0.5]], dtype=float)
    y_out = ai.parameter.Parameter(data=np.array(y_out_data, dtype=float), graph=graph)

    loss_fn = ai.loss.JSDivLoss(graph=graph)
    loss = loss_fn.forward(y_out, y_true_data)

    y_out_t = torch.tensor(np.array(y_out_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_true_t = torch.tensor(np.array(y_true_data, dtype=np.float64), dtype=torch.float64)
    y_mean_t = (y_out_t + y_true_t) / 2.0
    kl_1 = torch.nn.KLDivLoss(reduction="batchmean")(torch.log(y_mean_t), y_true_t)
    kl_2 = torch.nn.KLDivLoss(reduction="batchmean")(torch.log(y_mean_t), y_out_t)
    loss_t = 0.5 * (kl_1 + kl_2)

    np.testing.assert_allclose(loss.data.squeeze(), loss_t.detach().numpy(), rtol=1e-6, atol=1e-6)

    loss.backward()
    loss_t.backward()

    np.testing.assert_allclose(y_out.grad, y_out_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_test_loss_forward_backward():
    graph = ai.graph.ComputationalGraph()
    y_out_data = np.array([[1.0, -2.0], [3.0, 0.5]], dtype=float)
    y_out = ai.parameter.Parameter(data=np.array(y_out_data, dtype=float), graph=graph)

    loss_fn = ai.loss.TestLoss(graph=graph)
    loss = loss_fn.forward(y_out)

    y_out_t = torch.tensor(np.array(y_out_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    loss_t = y_out_t.sum() / y_out_t.shape[0]

    np.testing.assert_allclose(loss.data.squeeze(), loss_t.detach().numpy(), rtol=1e-6, atol=1e-6)

    loss.backward()
    loss_t.backward()

    np.testing.assert_allclose(y_out.grad, y_out_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)
