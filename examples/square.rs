use ndarray;

use optimize::nelder_mead as nm;

struct Square {
    value: f64
}
impl nm::Object for Square {
    fn evaluate(&self, x: &nm::VecXd) -> f64 {
        return (x[0] - self.value).powi(2);
    }
}

fn main() {
    let sq = Square{ value: 15.0 };

    let mut nms = nm::Simplex::new();
    nms.edge_length_threshold = 0.01;

    let guess: nm::VecXd = ndarray::array![1000.0];

    match nms.optimize(&sq, &guess) {
        Some(value) => println!("{}", value[0]),
        None => eprintln!("Couldn't optimize parameters!"),
    }
}
