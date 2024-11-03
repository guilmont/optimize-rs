// Provides Nelder-Mead minimization algorithm as in
// https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

use ndarray;

pub type VecXd = ndarray::Array1<f64>;

#[derive(Debug)]
struct Vertex {
    position: VecXd,
    weight: f64
}

impl Vertex {
    /// Initializes vertex object
    fn new(position: &VecXd, weight: f64) -> Self {
        return Self { position: position.clone(), weight };
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

pub struct Simplex {
    pub edge_length_threshold: f64,
    pub initial_edge_length: f64,
    pub max_iterations: usize,
}

/// Object trait implement function that must be evaluate for optimization
pub trait Object {
    fn evaluate(&self, params: &VecXd) -> f64;
}

impl Simplex {
    /// Initialize Nelder-Mead simplex with default values
    pub fn new() -> Self {
        return Self {
            edge_length_threshold: 0.001,
            initial_edge_length: 1.0,
            max_iterations: 5_000,
        };
    }

    /// Function called for optimize parameters for Object. Results a VecXd with optimal
    /// values if algorithm converged before maximum number of iterations was reached.
    pub fn optimize(&self, obj: &dyn Object, guesses: &VecXd) -> Option<VecXd> {
        // Some parameters needed by the method.Suggested values used
        const ALPHA: f64 = 1.0;
        const GAMMA: f64 = 2.0;
        const RHO: f64 = 0.5;
        const SIGMA: f64 = 0.5;
        let num_params: usize = guesses.len();

        // Simplex used for optimization
        let mut simplex = Vec::<Vertex>::new();

        // Initializing Simplex
        // The first vertex is the guess suggested by the user
        let mut vec = guesses.clone();
        simplex.push(Vertex::new(&vec, obj.evaluate(&vec)));

        // The simplex works with number of parameters + 1 dimension.
        // We need to set the others ourselves.
        for kk in 0..num_params {
            vec[kk] += self.initial_edge_length;
            simplex.push(Vertex::new(&vec, obj.evaluate(&vec)));
        }

        // Starting on the simplex method
        for _ in 0..self.max_iterations {
            // Check if the dimensions of simplex are acceptable
            if self.get_average_edge_size(&simplex) < self.edge_length_threshold {
                return Some(simplex[0].position.clone());
            }

            // Sorting from smallest to biggest vertex-weight.
            // On the option, we tell to consider NaN as equal
            simplex.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal));

            // Calculate the centroid for simplex with exception of last vertex
            let mut cm = VecXd::zeros(num_params);
            for vtx in &simplex[0..simplex.len() - 1] {
                cm += &vtx.position;
            }
            cm /= num_params as f64;

            // Calculate the reflection point between first and last vertex
            vec = &cm + ALPHA * (&cm - &simplex[num_params].position);
            let rf = Vertex::new(&vec, obj.evaluate(&vec));
            if (rf.weight < simplex[num_params - 1].weight) && (rf.weight > simplex[0].weight) {
                simplex[num_params] = rf;
            }
            // Calculate expansion point
            else if rf.weight < simplex[0].weight {
                vec = &cm + GAMMA * (&rf.position - &cm);
                let ex = Vertex::new(&vec, obj.evaluate(&vec));
                simplex[num_params] = if ex.weight < rf.weight { ex } else { rf };
            }
            // Calculate contraction point
            else {
                vec = &cm + RHO * (&simplex[num_params].position - &cm);
                let ct = Vertex::new(&vec, obj.evaluate(&vec));
                if ct.weight < simplex[num_params].weight {
                    simplex[num_params] = ct;
                }
                else { // Shrink
                    let pos_zero = simplex[0].position.clone();
                    for vtx in &mut simplex[1..] {
                        // Note that we don't need to recalculate the weights, they won't ever be used.
                        vtx.position = &pos_zero + SIGMA * (&vtx.position - &pos_zero);
                    }
                }
            }
        }
        // The simplex didn't convert before the maximum number of iterations
        return None;
    }

    /// Returns average longest distance between simplex vertices
    fn get_average_edge_size(&self, simplex: &Vec<Vertex>) -> f64 {
        let num_params: usize = simplex.len() - 1;
        let mut big = VecXd::from_vec(vec![f64::MIN; num_params]);
        let mut small = VecXd::from_vec(vec![f64::MAX; num_params]);

        for vtx in simplex.iter() {
            big.zip_mut_with(&vtx.position, |a, &b| *a = a.max(b));
            small.zip_mut_with(&vtx.position, |a, &b| *a = a.min(b));
        }

        return (&big - &small).iter().fold(f64::MIN, |a, &b| return a.max(b));
    }

} // impl Simplex
