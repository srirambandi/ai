import numpy as np
import torch

import ai.graph
import ai.parameter


# test all computational graph ops
def test_dot():
    graph = ai.graph.ComputationalGraph()
    x = ai.parameter.Parameter(data=np.array([1.0, 2.0, 3.0], dtype=float), graph=graph)
    y = ai.parameter.Parameter(data=np.array([0.5, -1.0, 2.0], dtype=float), graph=graph)

    out = graph.dot(x, y)

    x_t = torch.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(np.array([0.5, -1.0, 2.0], dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = x_t @ y_t

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    out.backward()
    out_t.backward()

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())
    np.testing.assert_allclose(y.grad, y_t.grad.detach().numpy())


def test_matmul():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, 2.0, -1.0], [0.0, 1.5, 3.0]], dtype=float)
    y_data = np.array([[2.0, -1.0, 0.5, 1.0], [1.0, 0.0, -2.0, 3.0], [0.5, 2.0, 1.0, -1.0]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)
    y = ai.parameter.Parameter(data=np.array(y_data, dtype=float), graph=graph)

    out = graph.matmul(x, y)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(np.array(y_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = x_t @ y_t

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(0).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())
    np.testing.assert_allclose(y.grad, y_t.grad.detach().numpy())


def test_matmul_broadcast():
    graph = ai.graph.ComputationalGraph()
    x_data = np.random.RandomState(0).randn(2, 3, 4).astype(float)
    y_data = np.random.RandomState(1).randn(4, 5).astype(float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)
    y = ai.parameter.Parameter(data=np.array(y_data, dtype=float), graph=graph)

    out = graph.matmul(x, y)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(np.array(y_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = x_t @ y_t

    np.testing.assert_allclose(out.data, out_t.detach().numpy(), rtol=1e-6, atol=1e-6)

    grad_out = np.random.RandomState(2).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y.grad, y_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_add():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.5]], dtype=float)
    y_data = np.array([[0.5, 2.0, -1.0], [1.0, -0.5, 2.5]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)
    y = ai.parameter.Parameter(data=np.array(y_data, dtype=float), graph=graph)

    out = graph.add(x, y)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(np.array(y_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = x_t + y_t

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(1).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())
    np.testing.assert_allclose(y.grad, y_t.grad.detach().numpy())


def test_add_broadcast():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.5]], dtype=float)
    y_data = np.array([0.5, 2.0, -1.0], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)
    y = ai.parameter.Parameter(data=np.array(y_data, dtype=float), graph=graph)

    out = graph.add(x, y)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(np.array(y_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = x_t + y_t

    np.testing.assert_allclose(out.data, out_t.detach().numpy(), rtol=1e-6, atol=1e-6)

    grad_out = np.random.RandomState(3).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y.grad, y_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_subtract():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.5]], dtype=float)
    y_data = np.array([[0.5, 2.0, -1.0], [1.0, -0.5, 2.5]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)
    y = ai.parameter.Parameter(data=np.array(y_data, dtype=float), graph=graph)

    out = graph.subtract(x, y)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(np.array(y_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = x_t - y_t

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(2).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())
    np.testing.assert_allclose(y.grad, y_t.grad.detach().numpy())


def test_subtract_broadcast():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.5]], dtype=float)
    y_data = np.array([0.5, 2.0, -1.0], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)
    y = ai.parameter.Parameter(data=np.array(y_data, dtype=float), graph=graph)

    out = graph.subtract(x, y)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(np.array(y_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = x_t - y_t

    np.testing.assert_allclose(out.data, out_t.detach().numpy(), rtol=1e-6, atol=1e-6)

    grad_out = np.random.RandomState(4).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y.grad, y_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_multiply():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.5]], dtype=float)
    y_data = np.array([[0.5, 2.0, -1.0], [1.0, -0.5, 2.5]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)
    y = ai.parameter.Parameter(data=np.array(y_data, dtype=float), graph=graph)

    out = graph.multiply(x, y)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(np.array(y_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = x_t * y_t

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(3).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())
    np.testing.assert_allclose(y.grad, y_t.grad.detach().numpy())


def test_multiply_broadcast():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.5]], dtype=float)
    y_data = np.array([0.5, 2.0, -1.0], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)
    y = ai.parameter.Parameter(data=np.array(y_data, dtype=float), graph=graph)

    out = graph.multiply(x, y)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(np.array(y_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = x_t * y_t

    np.testing.assert_allclose(out.data, out_t.detach().numpy(), rtol=1e-6, atol=1e-6)

    grad_out = np.random.RandomState(5).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y.grad, y_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_divide():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.5]], dtype=float)
    y_data = np.array([[0.5, 2.0, 1.5], [1.0, 0.5, 2.5]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)
    y = ai.parameter.Parameter(data=np.array(y_data, dtype=float), graph=graph)

    out = graph.divide(x, y, eps=1e-8)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(np.array(y_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = x_t / (y_t + 1e-8)

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(4).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())
    np.testing.assert_allclose(y.grad, y_t.grad.detach().numpy())


def test_divide_broadcast():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.5]], dtype=float)
    y_data = np.array([0.5, 2.0, 1.5], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)
    y = ai.parameter.Parameter(data=np.array(y_data, dtype=float), graph=graph)

    out = graph.divide(x, y, eps=1e-8)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(np.array(y_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = x_t / (y_t + 1e-8)

    np.testing.assert_allclose(out.data, out_t.detach().numpy(), rtol=1e-6, atol=1e-6)

    grad_out = np.random.RandomState(6).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y.grad, y_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_sum():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.5]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.sum(x, axis=1)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.sum(x_t, dim=1, keepdim=True)

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(5).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())


def test_power():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, -2.0]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.power(x, 3.0)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.pow(x_t, 3.0)

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(6).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())


def test_power_negative_exp():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, 2.0, 0.5], [1.5, 3.0, 2.0]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.power(x, -1.5)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.pow(x_t, -1.5)

    np.testing.assert_allclose(out.data, out_t.detach().numpy(), rtol=1e-6, atol=1e-6)

    grad_out = np.random.RandomState(7).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_log():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.log(x)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    safe_x_t = torch.where(x_t == 0, x_t + 1e-8, x_t)
    out_t = torch.log(safe_x_t)

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(7).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())


def test_conv1d():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[[1.0, -1.0, 2.0, 0.5, -0.5], [0.0, 1.0, -2.0, 3.0, 1.0]]], dtype=float)
    k_data = np.array([
        [[1.0, 0.5, -1.0], [0.0, 1.0, 0.5]],
        [[-1.0, 1.0, 0.5], [0.5, -0.5, 1.0]],
        [[0.25, -0.75, 1.5], [-0.5, 1.0, 0.0]],
    ], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)
    k = ai.parameter.Parameter(data=np.array(k_data, dtype=float), graph=graph)

    out = graph.conv1d(x, k, s=1, p=0)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    k_t = torch.tensor(np.array(k_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.nn.functional.conv1d(x_t, k_t, bias=None, stride=1, padding=0)

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(8).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(k.grad, k_t.grad.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_conv2d():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array(
        [[
            [[1.0, -1.0, 2.0, 0.5], [0.0, 1.0, -2.0, 3.0], [1.5, -0.5, 0.0, 2.0], [2.0, 1.0, -1.0, 0.5]],
            [[0.5, 2.0, -1.5, 1.0], [1.0, -0.5, 2.5, -2.0], [0.0, 1.0, -0.5, 1.5], [2.0, -1.0, 0.5, -0.5]],
        ]],
        dtype=float,
    )
    k_data = np.array(
        [
            [
                [[1.0, 0.5, -1.0], [0.0, 1.0, 0.5], [0.25, -0.75, 1.5]],
                [[-0.5, 1.0, 0.0], [1.0, -1.0, 0.5], [0.5, 0.0, -0.5]],
            ],
            [
                [[-1.0, 1.0, 0.5], [0.5, -0.5, 1.0], [0.25, 0.0, -0.25]],
                [[0.5, -0.5, 1.0], [1.5, -1.0, 0.0], [0.0, 0.5, -0.5]],
            ],
            [
                [[0.25, -0.75, 1.5], [-0.5, 1.0, 0.0], [1.0, -0.5, 0.25]],
                [[-0.25, 0.5, 0.75], [0.5, 0.0, -1.0], [1.0, 0.5, -0.5]],
            ],
        ],
        dtype=float,
    )
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)
    k = ai.parameter.Parameter(data=np.array(k_data, dtype=float), graph=graph)

    out = graph.conv2d(x, k, s=1, p=0)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    k_t = torch.tensor(np.array(k_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.nn.functional.conv2d(x_t, k_t, bias=None, stride=1, padding=0)

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(9).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(k.grad, k_t.grad.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_conv_transpose2d():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array(
        [[
            [[1.0, -1.0, 2.0], [0.0, 1.0, -2.0], [1.5, -0.5, 0.0]],
            [[0.5, 2.0, -1.5], [1.0, -0.5, 2.5], [0.0, 1.0, -0.5]],
        ]],
        dtype=float,
    )
    k_data = np.array(
        [
            [
                [[1.0, 0.5], [-1.0, 0.0]],
                [[0.5, -0.5], [1.0, 0.25]],
                [[-0.5, 1.0], [0.0, -1.0]],
            ],
            [
                [[-1.0, 0.5], [0.25, -0.75]],
                [[0.5, 1.0], [-0.5, 0.0]],
                [[1.0, -0.5], [0.5, 0.5]],
            ],
        ],
        dtype=float,
    )
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)
    k = ai.parameter.Parameter(data=np.array(k_data, dtype=float), graph=graph)

    out = graph.conv_transpose2d(x, k, s=1, p=0, a=0)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    k_t = torch.tensor(np.array(k_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.nn.functional.conv_transpose2d(x_t, k_t, bias=None, stride=1, padding=0, output_padding=0)

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(10).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(k.grad, k_t.grad.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_max_pool2d():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array(
        [[
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
        ]],
        dtype=float,
    )
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.max_pool2d(x, k=2, s=2, p=0)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.nn.functional.max_pool2d(x_t, kernel_size=2, stride=2, padding=0)

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(11).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())


def test_max_pool2d_ties():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array(
        [[
            [[1.0, 2.0], [2.0, 2.0]],
        ]],
        dtype=float,
    )
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.max_pool2d(x, k=2, s=2, p=0)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.nn.functional.max_pool2d(x_t, kernel_size=2, stride=2, padding=0)

    np.testing.assert_allclose(out.data, out_t.detach().numpy(), rtol=1e-6, atol=1e-6)

    grad_out = np.array([[[[1.0]]]], dtype=float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_dropout():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.5]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    p = 0.3
    np.random.seed(42)
    mask = (1.0 / (1.0 - p)) * np.random.binomial(1, 1.0 - p, size=x_data.shape)
    np.random.seed(42)

    out = graph.dropout(x, p=p)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    mask_t = torch.tensor(mask, dtype=torch.float64)
    out_t = mask_t * x_t

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(12).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())


def test_dropout_p1():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.5]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.dropout(x, p=1.0)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.nn.functional.dropout(x_t, p=1.0, training=True)

    np.testing.assert_allclose(out.data, out_t.detach().numpy(), rtol=1e-6, atol=1e-6)

    grad_out = np.random.RandomState(8).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_relu():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[-1.0, 2.0, -3.0], [4.0, -0.5, 1.5]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.relu(x)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.relu(x_t)

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(13).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())


def test_leaky_relu():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[-1.0, 2.0, -3.0], [4.0, -0.5, 1.5]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.leaky_relu(x, alpha=0.1)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.nn.functional.leaky_relu(x_t, negative_slope=0.1)

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(14).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())


def test_gelu():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[-1.0, 0.5, 2.0], [3.0, -0.5, 1.5]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.gelu(x)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.nn.functional.gelu(x_t, approximate="tanh")

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(15).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_sigmoid():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[-1.0, 0.5, 2.0], [3.0, -0.5, 1.5]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.sigmoid(x)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.sigmoid(x_t)

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(16).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())


def test_softmax():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, 2.0, 3.0], [0.5, -1.5, 2.5]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.softmax(x, axis=1)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.softmax(x_t, dim=1)

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(17).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_softmax_3d_axis():
    graph = ai.graph.ComputationalGraph()
    x_data = np.random.RandomState(9).randn(2, 3, 4).astype(float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.softmax(x, axis=-1)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.softmax(x_t, dim=-1)

    np.testing.assert_allclose(out.data, out_t.detach().numpy(), rtol=1e-6, atol=1e-6)

    grad_out = np.random.RandomState(10).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_tanh():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[-1.0, 0.5, 2.0], [3.0, -0.5, 1.5]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.tanh(x)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.tanh(x_t)

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(18).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())


def test_split():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    outs = graph.split(x, sections=2, axis=1)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    outs_t = torch.split(x_t, 2, dim=1)

    np.testing.assert_allclose(outs[0].data, outs_t[0].detach().numpy())
    np.testing.assert_allclose(outs[1].data, outs_t[1].detach().numpy())

    grad_out_0 = np.random.RandomState(19).randn(*outs[0].data.shape).astype(float)
    grad_out_1 = np.random.RandomState(20).randn(*outs[1].data.shape).astype(float)
    outs[1].grad = grad_out_1
    outs[0].backward(grad_out_0)

    outs_t[0].backward(torch.tensor(grad_out_0, dtype=torch.float64), retain_graph=True)
    outs_t[1].backward(torch.tensor(grad_out_1, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())


def test_getitem():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.getitem(x, (slice(None), slice(1, 3)))

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = x_t[:, 1:3]

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(21).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())


def test_cat():
    graph = ai.graph.ComputationalGraph()
    x1_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    x2_data = np.array([[5.0], [6.0]], dtype=float)
    x1 = ai.parameter.Parameter(data=np.array(x1_data, dtype=float), graph=graph)
    x2 = ai.parameter.Parameter(data=np.array(x2_data, dtype=float), graph=graph)

    out = graph.cat([x1, x2], axis=1)

    x1_t = torch.tensor(np.array(x1_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    x2_t = torch.tensor(np.array(x2_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = torch.cat([x1_t, x2_t], dim=1)

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(22).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x1.grad, x1_t.grad.detach().numpy())
    np.testing.assert_allclose(x2.grad, x2_t.grad.detach().numpy())


def test_transpose():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.transpose(x)

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = x_t.t()

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(23).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())


def test_reshape():
    graph = ai.graph.ComputationalGraph()
    x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    x = ai.parameter.Parameter(data=np.array(x_data, dtype=float), graph=graph)

    out = graph.reshape(x, (3, 2))

    x_t = torch.tensor(np.array(x_data, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out_t = x_t.reshape(3, 2)

    np.testing.assert_allclose(out.data, out_t.detach().numpy())

    grad_out = np.random.RandomState(24).randn(*out.data.shape).astype(float)
    out.backward(grad_out)
    out_t.backward(torch.tensor(grad_out, dtype=torch.float64))

    np.testing.assert_allclose(x.grad, x_t.grad.detach().numpy())
