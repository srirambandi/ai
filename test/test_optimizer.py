import numpy as np
import torch

import ai.graph
import ai.linear
import ai.loss
import ai.optimizer


def test_sgd_optimizer_step():
    graph = ai.graph.ComputationalGraph()
    model = ai.linear.Linear(2, 2, bias=True, graph=graph)
    model.weight.data = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=float)
    model.bias.data = np.array([[0.05, -0.05]], dtype=float)

    x_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    y_true_data = np.array([[0.5, -1.0], [1.5, 2.0]], dtype=float)
    x = np.array(x_data, dtype=float)

    y_out = model.forward(x)
    loss_fn = ai.loss.MSELoss(graph=graph)
    loss = loss_fn.forward(y_out, y_true_data)
    loss.backward()

    opt = ai.optimizer.SGD([model.weight, model.bias], lr=0.1, momentum=0.0)
    opt.step()

    torch_model = torch.nn.Linear(2, 2, bias=True, dtype=torch.float64)
    with torch.no_grad():
        torch_model.weight.copy_(torch.tensor(np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float64)))
        torch_model.bias.copy_(torch.tensor(np.array([0.05, -0.05], dtype=np.float64)))

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64)
    y_true_t = torch.tensor(np.array(y_true_data, dtype=np.float64), dtype=torch.float64)
    y_out_t = torch_model(x_t)
    loss_t = torch.nn.functional.mse_loss(y_out_t, y_true_t, reduction="mean")
    loss_t.backward()

    torch_opt = torch.optim.SGD(torch_model.parameters(), lr=0.1, momentum=0.0)
    torch_opt.step()

    np.testing.assert_allclose(model.weight.data, torch_model.weight.detach().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(model.bias.data.squeeze(), torch_model.bias.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_sgd_optimizer_step_with_momentum():
    graph = ai.graph.ComputationalGraph()
    model = ai.linear.Linear(2, 2, bias=True, graph=graph)
    model.weight.data = np.array([[0.2, -0.1], [0.0, 0.3]], dtype=float)
    model.bias.data = np.array([[0.1, -0.2]], dtype=float)

    x_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    y_true_data = np.array([[0.5, -1.0], [1.5, 2.0]], dtype=float)
    x = np.array(x_data, dtype=float)

    y_out = model.forward(x)
    loss_fn = ai.loss.MSELoss(graph=graph)
    loss = loss_fn.forward(y_out, y_true_data)
    loss.backward()

    opt = ai.optimizer.SGD([model.weight, model.bias], lr=0.1, momentum=0.9)
    opt.step()

    torch_model = torch.nn.Linear(2, 2, bias=True, dtype=torch.float64)
    with torch.no_grad():
        torch_model.weight.copy_(torch.tensor(np.array([[0.2, -0.1], [0.0, 0.3]], dtype=np.float64)))
        torch_model.bias.copy_(torch.tensor(np.array([0.1, -0.2], dtype=np.float64)))

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64)
    y_true_t = torch.tensor(np.array(y_true_data, dtype=np.float64), dtype=torch.float64)
    y_out_t = torch_model(x_t)
    loss_t = torch.nn.functional.mse_loss(y_out_t, y_true_t, reduction="mean")
    loss_t.backward()

    torch_opt = torch.optim.SGD(torch_model.parameters(), lr=0.1, momentum=0.9)
    torch_opt.step()

    np.testing.assert_allclose(model.weight.data, torch_model.weight.detach().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(model.bias.data.squeeze(), torch_model.bias.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_adam_optimizer_step():
    graph = ai.graph.ComputationalGraph()
    model = ai.linear.Linear(2, 2, bias=True, graph=graph)
    model.weight.data = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=float)
    model.bias.data = np.array([[0.05, -0.05]], dtype=float)

    x_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    y_true_data = np.array([[0.5, -1.0], [1.5, 2.0]], dtype=float)
    x = np.array(x_data, dtype=float)

    y_out = model.forward(x)
    loss_fn = ai.loss.MSELoss(graph=graph)
    loss = loss_fn.forward(y_out, y_true_data)
    loss.backward()

    opt = ai.optimizer.Adam([model.weight, model.bias], lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)
    opt.step()

    torch_model = torch.nn.Linear(2, 2, bias=True, dtype=torch.float64)
    with torch.no_grad():
        torch_model.weight.copy_(torch.tensor(np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float64)))
        torch_model.bias.copy_(torch.tensor(np.array([0.05, -0.05], dtype=np.float64)))

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64)
    y_true_t = torch.tensor(np.array(y_true_data, dtype=np.float64), dtype=torch.float64)
    y_out_t = torch_model(x_t)
    loss_t = torch.nn.functional.mse_loss(y_out_t, y_true_t, reduction="mean")
    loss_t.backward()

    torch_opt = torch.optim.Adam(torch_model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    torch_opt.step()

    np.testing.assert_allclose(model.weight.data, torch_model.weight.detach().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(model.bias.data.squeeze(), torch_model.bias.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_adagrad_optimizer_step():
    graph = ai.graph.ComputationalGraph()
    model = ai.linear.Linear(2, 2, bias=True, graph=graph)
    model.weight.data = np.array([[0.15, -0.05], [0.2, 0.25]], dtype=float)
    model.bias.data = np.array([[0.0, 0.1]], dtype=float)

    x_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    y_true_data = np.array([[0.5, -1.0], [1.5, 2.0]], dtype=float)
    x = np.array(x_data, dtype=float)

    y_out = model.forward(x)
    loss_fn = ai.loss.MSELoss(graph=graph)
    loss = loss_fn.forward(y_out, y_true_data)
    loss.backward()

    opt = ai.optimizer.Adagrad([model.weight, model.bias], lr=0.1, eps=1e-8)
    opt.step()

    torch_model = torch.nn.Linear(2, 2, bias=True, dtype=torch.float64)
    with torch.no_grad():
        torch_model.weight.copy_(torch.tensor(np.array([[0.15, -0.05], [0.2, 0.25]], dtype=np.float64)))
        torch_model.bias.copy_(torch.tensor(np.array([0.0, 0.1], dtype=np.float64)))

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64)
    y_true_t = torch.tensor(np.array(y_true_data, dtype=np.float64), dtype=torch.float64)
    y_out_t = torch_model(x_t)
    loss_t = torch.nn.functional.mse_loss(y_out_t, y_true_t, reduction="mean")
    loss_t.backward()

    torch_opt = torch.optim.Adagrad(torch_model.parameters(), lr=0.1, eps=1e-8)
    torch_opt.step()

    np.testing.assert_allclose(model.weight.data, torch_model.weight.detach().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(model.bias.data.squeeze(), torch_model.bias.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_adadelta_optimizer_step():
    graph = ai.graph.ComputationalGraph()
    model = ai.linear.Linear(2, 2, bias=True, graph=graph)
    model.weight.data = np.array([[0.05, -0.15], [0.25, 0.35]], dtype=float)
    model.bias.data = np.array([[0.0, -0.1]], dtype=float)

    x_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    y_true_data = np.array([[0.5, -1.0], [1.5, 2.0]], dtype=float)
    x = np.array(x_data, dtype=float)

    y_out = model.forward(x)
    loss_fn = ai.loss.MSELoss(graph=graph)
    loss = loss_fn.forward(y_out, y_true_data)
    loss.backward()

    opt = ai.optimizer.Adadelta([model.weight, model.bias], rho=0.95, eps=1e-8)
    opt.step()

    torch_model = torch.nn.Linear(2, 2, bias=True, dtype=torch.float64)
    with torch.no_grad():
        torch_model.weight.copy_(torch.tensor(np.array([[0.05, -0.15], [0.25, 0.35]], dtype=np.float64)))
        torch_model.bias.copy_(torch.tensor(np.array([0.0, -0.1], dtype=np.float64)))

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64)
    y_true_t = torch.tensor(np.array(y_true_data, dtype=np.float64), dtype=torch.float64)
    y_out_t = torch_model(x_t)
    loss_t = torch.nn.functional.mse_loss(y_out_t, y_true_t, reduction="mean")
    loss_t.backward()

    torch_opt = torch.optim.Adadelta(torch_model.parameters(), rho=0.95, eps=1e-8, lr=1.0)
    torch_opt.step()

    np.testing.assert_allclose(model.weight.data, torch_model.weight.detach().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(model.bias.data.squeeze(), torch_model.bias.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_rmsprop_optimizer_step():
    graph = ai.graph.ComputationalGraph()
    model = ai.linear.Linear(2, 2, bias=True, graph=graph)
    model.weight.data = np.array([[0.12, -0.22], [0.18, 0.28]], dtype=float)
    model.bias.data = np.array([[0.02, -0.03]], dtype=float)

    x_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    y_true_data = np.array([[0.5, -1.0], [1.5, 2.0]], dtype=float)
    x = np.array(x_data, dtype=float)

    y_out = model.forward(x)
    loss_fn = ai.loss.MSELoss(graph=graph)
    loss = loss_fn.forward(y_out, y_true_data)
    loss.backward()

    opt = ai.optimizer.RMSprop([model.weight, model.bias], lr=0.01, alpha=0.99, eps=1e-8)
    opt.step()

    torch_model = torch.nn.Linear(2, 2, bias=True, dtype=torch.float64)
    with torch.no_grad():
        torch_model.weight.copy_(torch.tensor(np.array([[0.12, -0.22], [0.18, 0.28]], dtype=np.float64)))
        torch_model.bias.copy_(torch.tensor(np.array([0.02, -0.03], dtype=np.float64)))

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64)
    y_true_t = torch.tensor(np.array(y_true_data, dtype=np.float64), dtype=torch.float64)
    y_out_t = torch_model(x_t)
    loss_t = torch.nn.functional.mse_loss(y_out_t, y_true_t, reduction="mean")
    loss_t.backward()

    torch_opt = torch.optim.RMSprop(torch_model.parameters(), lr=0.01, alpha=0.99, eps=1e-8, momentum=0.0)
    torch_opt.step()

    np.testing.assert_allclose(model.weight.data, torch_model.weight.detach().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(model.bias.data.squeeze(), torch_model.bias.detach().numpy(), rtol=1e-6, atol=1e-6)
