#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfusionMatrix {
    classes: usize,
    counts: Vec<usize>,
}

impl ConfusionMatrix {
    pub fn new(classes: usize) -> Self {
        Self {
            classes,
            counts: vec![0; classes * classes],
        }
    }

    pub fn record(&mut self, actual: usize, predicted: usize) {
        self.counts[actual * self.classes + predicted] += 1;
    }

    pub fn merge(&mut self, other: &ConfusionMatrix) {
        assert_eq!(self.classes, other.classes);
        for (a, b) in self.counts.iter_mut().zip(&other.counts) {
            *a += b;
        }
    }

    pub fn count(&self, actual: usize, predicted: usize) -> usize {
        self.counts[actual * self.classes + predicted]
    }

    pub fn total(&self) -> usize {
        self.counts.iter().sum()
    }

    pub fn correct(&self) -> usize {
        (0..self.classes).map(|c| self.count(c, c)).sum()
    }

    pub fn accuracy(&self) -> f32 {
        ratio(self.correct(), self.total())
    }

    pub fn recall(&self, class: usize) -> f32 {
        ratio(
            self.count(class, class),
            (0..self.classes).map(|p| self.count(class, p)).sum(),
        )
    }

    pub fn precision(&self, class: usize) -> f32 {
        ratio(
            self.count(class, class),
            (0..self.classes).map(|a| self.count(a, class)).sum(),
        )
    }

    /// Harmonic mean of precision and recall for `class` (0 when both are 0).
    pub fn f1_score(&self, class: usize) -> f32 {
        let (p, r) = (self.precision(class), self.recall(class));
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// Unweighted mean of the per-class precisions.
    pub fn macro_precision(&self) -> f32 {
        self.macro_average(Self::precision)
    }

    /// Unweighted mean of the per-class recalls ("balanced accuracy").
    pub fn macro_recall(&self) -> f32 {
        self.macro_average(Self::recall)
    }

    /// Unweighted mean of the per-class F1 scores. Unlike accuracy, every class counts
    /// equally, so a model cannot score well by ignoring a rare class.
    pub fn macro_f1(&self) -> f32 {
        self.macro_average(Self::f1_score)
    }

    /// Matthews correlation coefficient, generalised to several classes (Gorodkin, 2004).
    /// 1 is a perfect prediction, 0 no better than chance, −1 total disagreement.
    pub fn mcc(&self) -> f32 {
        let n = self.classes;
        let total = self.total() as f64;
        let correct = self.correct() as f64;
        let actual: Vec<f64> = (0..n)
            .map(|a| (0..n).map(|p| self.count(a, p)).sum::<usize>() as f64)
            .collect();
        let predicted: Vec<f64> = (0..n)
            .map(|p| (0..n).map(|a| self.count(a, p)).sum::<usize>() as f64)
            .collect();
        let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(a, b)| a * b).sum::<f64>();
        let numerator = correct * total - dot(&actual, &predicted);
        let denominator = ((total * total - dot(&predicted, &predicted))
            * (total * total - dot(&actual, &actual)))
        .sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            (numerator / denominator) as f32
        }
    }

    fn macro_average(&self, metric: fn(&Self, usize) -> f32) -> f32 {
        if self.classes == 0 {
            return 0.0;
        }
        (0..self.classes).map(|c| metric(self, c)).sum::<f32>() / self.classes as f32
    }

    pub fn render(&self, labels: &[String]) -> String {
        let name = |i: usize| labels.get(i).map_or_else(|| i.to_string(), |s| s.clone());
        let mut out = format!("{:<20}", "actual \\ predicted");
        for p in 0..self.classes {
            out += &format!("{:>8}", name(p));
        }
        out += &format!("{:>10}\n", "recall");
        for a in 0..self.classes {
            out += &format!("{:<20}", name(a));
            for p in 0..self.classes {
                out += &format!("{:>8}", self.count(a, p));
            }
            out += &format!("{:>9.1}%\n", self.recall(a) * 100.0);
        }
        out += &format!("{:<20}", "precision");
        for p in 0..self.classes {
            out += &format!("{:>7.1}%", self.precision(p) * 100.0);
        }
        out += &format!("\n{:<20}", "f1");
        for c in 0..self.classes {
            out += &format!("{:>7.1}%", self.f1_score(c) * 100.0);
        }
        out += &format!(
            "\n\naccuracy: {:.2}% ({} / {})\n",
            self.accuracy() * 100.0,
            self.correct(),
            self.total()
        );
        out += &format!(
            "macro f1: {:.2}%   balanced accuracy: {:.2}%   mcc: {:.3}\n",
            self.macro_f1() * 100.0,
            self.macro_recall() * 100.0,
            self.mcc()
        );
        out
    }
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_accuracy_precision_recall() {
        let mut cm = ConfusionMatrix::new(2);
        cm.record(0, 0);
        cm.record(0, 0);
        cm.record(0, 1);
        cm.record(1, 1);
        assert_eq!(cm.accuracy(), 0.75);
        assert!((cm.recall(0) - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(cm.precision(1), 0.5);

        let mut other = ConfusionMatrix::new(2);
        other.record(1, 0);
        cm.merge(&other);
        assert_eq!(cm.total(), 5);
        assert!(
            cm.render(&["a".into(), "b".into()])
                .contains("accuracy: 60.00%")
        );
    }

    #[test]
    fn computes_f1_and_macro_averages() {
        let mut cm = ConfusionMatrix::new(2);
        for (actual, predicted, times) in [(0, 0, 2), (0, 1, 1), (1, 1, 1)] {
            for _ in 0..times {
                cm.record(actual, predicted);
            }
        }
        // class 0: precision 1, recall 2/3; class 1: precision 1/2, recall 1.
        assert!((cm.f1_score(0) - 0.8).abs() < 1e-6);
        assert!((cm.f1_score(1) - 2.0 / 3.0).abs() < 1e-6);
        assert!((cm.macro_f1() - (0.8 + 2.0 / 3.0) / 2.0).abs() < 1e-6);
        assert!((cm.macro_recall() - (2.0 / 3.0 + 1.0) / 2.0).abs() < 1e-6);
        assert!((cm.macro_precision() - 0.75).abs() < 1e-6);
        let empty = ConfusionMatrix::new(3);
        assert_eq!(
            (empty.f1_score(0), empty.macro_f1(), empty.mcc()),
            (0.0, 0.0, 0.0)
        );
    }

    #[test]
    fn mcc_matches_the_binary_formula_and_its_extremes() {
        // Binary MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)).
        let (tp, tn, fp, fun) = (6usize, 3usize, 1usize, 2usize);
        let mut cm = ConfusionMatrix::new(2);
        let mut add = |a, p, n| (0..n).for_each(|_| cm.record(a, p));
        add(1, 1, tp);
        add(0, 0, tn);
        add(0, 1, fp);
        add(1, 0, fun);
        let (tp, tn, fp, fun) = (tp as f64, tn as f64, fp as f64, fun as f64);
        let expected =
            (tp * tn - fp * fun) / ((tp + fp) * (tp + fun) * (tn + fp) * (tn + fun)).sqrt();
        assert!((cm.mcc() as f64 - expected).abs() < 1e-6);

        let mut perfect = ConfusionMatrix::new(3);
        (0..3).for_each(|c| perfect.record(c, c));
        assert!((perfect.mcc() - 1.0).abs() < 1e-6);
        // Always predicting one class carries no information.
        let mut constant = ConfusionMatrix::new(3);
        (0..3).for_each(|c| constant.record(c, 0));
        assert_eq!(constant.mcc(), 0.0);
    }
}
