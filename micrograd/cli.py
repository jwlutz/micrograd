"""Command-line interface for the micrograd demos.

Train a network on a 1-D function or a 2-D classification task, on the micrograd
engine (with optional animation) or on PyTorch / JAX for comparison.

Examples:
  python -m micrograd.cli --task sin
  python -m micrograd.cli --task sin --loss mae --gif assets/learn_sin.gif
  python -m micrograd.cli --task moons --loss logistic --gif assets/moons.gif
  python -m micrograd.cli --task spiral --framework pytorch
  python -m micrograd.cli --task sin --framework jax --epochs 1500 --lr 0.05
  python -m micrograd.cli --list
"""

import argparse
import math

from micrograd import classify, frameworks, learn

# 1-D regression targets: name -> (function, domain)
REG_TASKS = {
    "sin":    (math.sin, (-math.pi, math.pi)),
    "abs":    (abs, (-2.0, 2.0)),
    "quad":   (lambda x: x * x, (-2.0, 2.0)),
    "step":   (lambda x: 1.0 if x > 0 else -1.0, (-2.0, 2.0)),
    "damped": (lambda x: math.exp(-0.3 * abs(x)) * math.sin(3 * x), (-math.pi, math.pi)),
}
CLF_TASKS = ("moons", "spiral")

REG_LOSSES = ("mse", "mae", "huber")
CLF_LOSSES = ("hinge", "svm", "max_margin", "squared_hinge", "logistic")  # svm/max_margin == hinge (Karpathy's)

# per-task defaults: (epochs, lr, hidden)
DEFAULTS = {
    "sin":    (2000, 0.08, (12,)),
    "abs":    (2500, 0.08, (16,)),
    "quad":   (2000, 0.05, (12,)),
    "step":   (3000, 0.05, (16,)),
    "damped": (2500, 0.08, (16,)),
    "moons":  (120, 0.10, (16, 16)),
    "spiral": (1500, 0.30, (16, 16)),
}


def parse_hidden(s):
    return tuple(int(v) for v in s.split(",") if v.strip())


def main(argv=None):
    p = argparse.ArgumentParser(description="train and visualize a micrograd network")
    p.add_argument("--task", choices=list(REG_TASKS) + list(CLF_TASKS),
                   help="function (sin/abs/quad/step/damped) or dataset (moons/spiral)")
    p.add_argument("--framework", default="micrograd", choices=["micrograd", "pytorch", "jax"],
                   help="engine to train on (animation only available for micrograd)")
    p.add_argument("--loss", default=None,
                   help="regression: mse/mae/huber ; classification: hinge/squared_hinge/logistic")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="step size")
    p.add_argument("--hidden", type=str, default=None, help="comma-separated, e.g. 16,16")
    p.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization (classification)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--noise", type=float, default=0.10, help="dataset noise (moons/spiral)")
    p.add_argument("--gif", type=str, default=None, help="save animation to this path (micrograd only)")
    p.add_argument("--log-every", type=int, default=25, help="print progress every N epochs (0 = silent)")
    p.add_argument("--list", action="store_true", help="list available tasks and losses, then exit")
    args = p.parse_args(argv)

    if args.list:
        print("regression tasks:    ", ", ".join(REG_TASKS))
        print("classification tasks:", ", ".join(CLF_TASKS))
        print("regression losses:   ", ", ".join(REG_LOSSES))
        print("classification losses:", ", ".join(CLF_LOSSES))
        return
    if not args.task:
        p.error("--task is required (or use --list)")

    epochs, lr, hidden = DEFAULTS[args.task]
    if args.epochs is not None:
        epochs = args.epochs
    if args.lr is not None:
        lr = args.lr
    if args.hidden:
        hidden = parse_hidden(args.hidden)

    is_reg = args.task in REG_TASKS
    if is_reg:
        loss = args.loss or "mse"
        if loss not in REG_LOSSES:
            p.error(f"--loss {loss!r} is not a regression loss; choose from {REG_LOSSES}")
    else:
        loss = args.loss or "hinge"
        if loss not in CLF_LOSSES:
            p.error(f"--loss {loss!r} is not a classification loss; choose from {CLF_LOSSES}")

    print(f"task={args.task}  framework={args.framework}  loss={loss}  "
          f"epochs={epochs}  lr={lr}  hidden={hidden}\n")

    if args.framework != "micrograd" and args.gif:
        print("note: --gif is ignored; animation is only available on the micrograd engine\n")

    if is_reg:
        target, domain = REG_TASKS[args.task]
        if args.framework == "micrograd":
            history = learn.fit_function(target=target, domain=domain, hidden=hidden[0],
                                         epochs=epochs, lr=lr, seed=args.seed,
                                         log_every=args.log_every, loss=loss)
            print(f"\nfinal loss {history['losses'][-1]:.5f}")
            if args.gif:
                learn.animate_fit(history, args.gif)
                print(f"saved {args.gif}")
        else:
            xs = [domain[0] + (domain[1] - domain[0]) * i / 39 for i in range(40)]
            ys = [target(x) for x in xs]
            run = frameworks.torch_regression if args.framework == "pytorch" else frameworks.jax_regression
            run(xs, ys, hidden, epochs, lr, args.seed, args.log_every, loss)
    else:
        maker = classify.make_moons if args.task == "moons" else classify.make_spiral
        pts, labels = maker(n=120, noise=args.noise, seed=args.seed)
        if args.framework == "micrograd":
            history = classify.fit_classifier(pts, labels, hidden=hidden, epochs=epochs, lr=lr,
                                              alpha=args.alpha, seed=args.seed,
                                              log_every=args.log_every, loss=loss)
            print(f"\nfinal acc {history['accs'][-1] * 100:.0f}%")
            if args.gif:
                classify.animate_classifier(history, args.gif)
                print(f"saved {args.gif}")
        else:
            run = frameworks.torch_classification if args.framework == "pytorch" else frameworks.jax_classification
            run(pts, labels, hidden, epochs, lr, alpha=args.alpha, seed=args.seed,
                log_every=args.log_every, loss=loss)


if __name__ == "__main__":
    main()
