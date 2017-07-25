class ANN {

    constructor(config) {
        this.initialize(config);
        this.buildNet();
    }

    initialize(config) {
        var CONFIG = {
            layers: {
                input: 2,
                hidden: 100,
                output: 1
            },
            depth: 4,
            rate: 0.1,
            bias: true
        };
        this.config = CONFIG;
        this.layers = {
            input: [],
            hidden: [],
            output: []
        };
    }

    buildNet() {
        //Build Neurons
        for (var i = 0; i < this.config.layers.input; i++) {
            this.layers.input.push(new Neuron());
        }

        for (var i = 0; i < this.config.layers.output; i++) {
            this.layers.output.push(new Neuron());
        }

        for (var i = 0; i < this.config.depth; i++) {
            var layer = [];
            for (var j = 0; j < this.config.layers.hidden; j++) {
                layer.push(new Neuron());
            }
            this.layers.hidden.push(layer);
        }

        //Add Synapses
        var synapse;
        for (var i = 0; i < this.config.layers.input; i++) {
            for (var j = 0; j < this.config.layers.hidden; j++) {
                synapse = new Synapse(this.layers.hidden[0][j], this.layers.input[i]);
                this.layers.input[i].addSynapse(synapse, "next");
                this.layers.hidden[0][j].addSynapse(synapse, "previous");
            }
        }

        for (var i = 0; i < this.layers.hidden.length - 1; i++) {
            for (var j in this.layers.hidden[i]) {
                for (var k in this.layers.hidden[i + 1]) {
                    synapse = new Synapse(this.layers.hidden[i + 1][k], this.layers.hidden[i][j]);
                    this.layers.hidden[i][j].addSynapse(synapse, "next");
                    this.layers.hidden[i + 1][k].addSynapse(synapse, "previous");
                }
            }
        }

        for (var i = 0; i < this.layers.hidden[this.layers.hidden.length - 1].length; i++) {
            for (var j in this.layers.output) {
                synapse = new Synapse(this.layers.output[j], this.layers.hidden[this.layers.hidden.length - 1][i]);
                this.layers.hidden[this.layers.hidden.length - 1][i].addSynapse(synapse, "next");
                this.layers.output[j].addSynapse(synapse, "previous");
            }
        }

        if (this.config.bias) {
            var temp = new Neuron(1, true);
            var synapse;
            for (var i = 0; i < this.config.layers.hidden; i++) {
                synapse = new Synapse(this.layers.hidden[0][i], temp);
                temp.addSynapse(synapse, "next");
            }
            this.layers.input.push(temp);

            for (var i = 0; i < this.config.depth - 1; i++) {
                temp = new Neuron(1, true);
                for (var j = 0; j < this.config.layers.hidden; j++) {
                    synapse = new Synapse(this.layers.hidden[i + 1][j], temp);
                    temp.addSynapse(synapse, "next");
                }
                this.layers.hidden[i].push(temp);
            }

            temp = new Neuron(1, true);
            for (var i = 0; i < this.config.layers.output; i++) {
                synapse = new Synapse(this.layers.output[i], temp);
                temp.addSynapse(synapse, "next");
            }
            this.layers.hidden[this.config.depth - 1].push(temp);
        }
    }

    train(arr, res, num) {
        if (arr.length != this.config.layers.input) {
            //error message: incorrect number of input neurons.
            return;
        }

        num = num || 2000;

        for (var i = 0; i < num; i++) {
            var pred = this.forward(arr);
            this.backward(res);
            this.clearNeurons();
        }
        console.log("Done training: " + arr);
    }

    predict(arr, res) {
        var est = this.forward(arr);
        this.clearNeurons();
        console.log(JSON.stringify(arr) + " -> " + JSON.stringify(est) + " should be (" + JSON.stringify(res) + ")");
        //        return est;
    }

    forward(input) {

        for (var i in input) {
            this.layers.input[i].setVal(input[i]);
        }

        for (var i in this.layers.input) {
            this.layers.input[i].forward();
        }
        for (var i = 0; i < this.layers.hidden.length; i++) {
            for (var j = 0; j < this.layers.hidden[i].length; j++) {
                this.layers.hidden[i][j].forward();
            }
        }
        var tmp = [];
        for (var i in this.layers.output) {
            tmp.push(this.layers.output[i].getVal());
        }
        return tmp;
    }

    backward(real) {

        for (var i in real) {
            this.layers.output[i].shouldBe(real[i]);
        }

        for (var i in this.layers.output) {
            this.layers.output[i].backward(this.config.rate);
        }

        for (var i = this.config.depth - 1; i >= 0; i--) {
            for (var j = 0; j < this.config.layers.hidden; j++) {
                this.layers.hidden[i][j].backward(this.config.rate);
            }
        }
    }

    saveNet() {
        //output JSON file with saved state
    }

    printNet() {
        var layers = [];
        var synapse = [];
        var temp = [];
        layers.push([]);
        synapse.push([]);
        for (var i in this.layers.input) {
            layers[0].push(this.layers.input[i].getVal());
            temp = [];
            for (var j in this.layers.input[i].synapse.next) {
                temp.push(this.layers.input[i].synapse.next[j].weight);
            }
            synapse[0].push(temp);
        }
        console.log(JSON.stringify(layers[0]));
        console.log(JSON.stringify(synapse[0]));
        console.log(this);

    }

    clearNeurons() {
        for (var i in this.layers.hidden) {
            for (var j in this.layers.hidden[i]) {
                this.layers.hidden[i][j].reset();
            }
        }
        for (var i in this.layers.output) {
            this.layers.output[i].reset();
        }
    }

}



class Neuron {
    constructor(v, bias) {
        this.val = v || 0;
        this.shouldbe = 0;
        this.synapse = {
            next: [],
            previous: []
        }
        this.id = Math.random().toString(36).slice(2);
        this.bias = !(!bias);
    }

    sigmoid(n) {
        return n;
        //        return 1 / (1 + Math.pow(Math.E, -n));
    }

    sigmoidPrime(n) {
        return 1;
        //        return this.sigmoid(n) * (1 - this.sigmoid(n));
    }

    logit(n) {
        return Math.log(n / (1 - n));
    }

    reset() {
        this.val = 0;
    }

    addSynapse(obj, dir) {
        this.synapse[dir].push(obj);
    }

    getVal() {
        return this.val;
    }

    shouldBe(val) {
        this.shouldbe = val;
    }

    forward(num) {
        if (num) {
            this.val += num;
        }
        for (var i in this.synapse.next) {
            var pass = this.synapse.previous.length == 0 ? this.val : this.sigmoid(this.val);
            this.synapse.next[i].forward(this.bias ? 1 : pass);
        }
    }

    backward(rate) {
        //        var sigP = this.sigmoidPrime(this.val);

        var error = rate * (-this.val + this.shouldbe);

        for (var i in this.synapse.previous) {
            this.synapse.previous[i].backward(this.val, error);
        }
    }

    setVal(num) {
        this.val = num;
    }

    addVal(num) {
        if (num && this.synapse.previous.length > 0) {
            this.val += num;
        }
    }
}


class Synapse {
    constructor(next, previous) {
        this.weight = Math.random();
        this.next = next;
        this.previous = previous;
        this.id = Math.random().toString(36).slice(2);
    }

    setWeight(num) {
        this.weight = num;
    }

    forward(val) {
        this.next.addVal(this.weight * val);
    }

    backward(val, err) {
        var portion = this.previous.getVal() / val * err;

        this.weight += portion / this.previous.getVal();
        this.previous.shouldBe(this.previous.getVal() + portion);
    }
}


//module.exports = ANN;
