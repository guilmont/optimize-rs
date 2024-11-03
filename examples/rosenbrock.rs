use ::optimize::nelder_mead as nm;

struct Rosenbrock {
    a: f64,
    b: f64,
}

impl nm::Object for Rosenbrock {
    fn evaluate(&self, x: &nm::VecXd) -> f64 {
        return (self.a - x[0]).powi(2) + self.b * (x[1] - x[0] * x[0]).powi(2);
    }
}


fn main() {
    let rb = Rosenbrock {a: 1.0, b: 100.0 };

    let mut nms = nm::Simplex::new();
    nms.edge_length_threshold = 0.001;

    match nms.optimize(&rb, &nm::VecXd::from_vec(vec![50.0,-50.0])) {
        Some(res) => println!("{}, {}", res[0], res[1]),
        None => eprintln!("ERROR: Couldn't optimize parameters"),
    }
}
