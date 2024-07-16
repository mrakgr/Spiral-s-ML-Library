import { LitElement, PropertyValueMap, PropertyValues, css, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { map } from 'lit/directives/map.js';
import { createRef, Ref, ref } from 'lit/directives/ref.js';
import { io } from 'socket.io-client'
import * as echarts from 'echarts';
import { assert, assert_tag_is_never } from './utils';
import { serialize } from '@shoelace-style/shoelace/dist/utilities/form.js';

type Leduc_Train_Label = string
type Leduc_Train_Serie = {
    name: string
    data: number[]
}
type Public_State = {
    training_iterations: number
}
type UI_Effect = ["GraphAddItem", [Leduc_Train_Label[], Leduc_Train_Serie[]]]
type Train_Events = ["Train", Public_State]

class TrainElement extends LitElement {
    dispatch_train_event = (detail: Train_Events) => {
        this.dispatchEvent(new CustomEvent('train', { bubbles: true, composed: true, detail }))
    }
}

@customElement('leduc-train-ui')
class Leduc_Train_UI extends TrainElement {
    @property({ type: Object }) state: Public_State = {
        training_iterations: 100
    }

    socket = io('/leduc_train')
    constructor() {
        super()
        this.socket.on('update', (x: [Public_State, UI_Effect[]]) => {
            this.state = x[0];
            this.process_effects(x[1])
        });
        this.addEventListener('train', (ev) => {
            ev.stopPropagation();
            this.socket.emit('update', (ev as CustomEvent<Train_Events>).detail);
        })
    }

    graph_ref = createRef<Training_Chart>()
    process_effects(l: UI_Effect[]) {
        l.forEach(l => {
            const [tag, data] = l
            switch (tag) {
                case 'GraphAddItem': {
                    this.graph_ref.value?.add_item(...data)
                    break;
                }
                default: assert_tag_is_never(tag);
            }
        })
    }

    static styles = css`
        :host {
            padding: 0 50px;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            height: 100%;
            width: 100%;
        }

        training-chart {
            height: 80%;
            width: 80%;
        }

        sl-input, sl-button {
            width: fit-content;
        }
    `
    render() {
        return html`
            <training-chart ${ref(this.graph_ref)}></training-chart>
            <sl-input
                type="number" name="iters" min="0" step="100" 
                value=${this.state.training_iterations}
                @sl-input=${(x: any) => this.state = { ...this.state, training_iterations: parseInt(x.target.value) || this.state.training_iterations }}
                label="Number of iterations:"
                ></sl-input>
            <br/>
            <sl-button variant="primary" @click=${this.on_train}>Train</sl-button>
            `
    }

    on_train() {
        this.dispatch_train_event(["Train", this.state])
    }
}

const assert_chart_data = (labels: Leduc_Train_Label[], series: Leduc_Train_Serie[]) => {
    assert(series.every(x => labels.length === x.data.length), "The length of the labels array does not match that of the series data.");
}
const update_chart = (chart: echarts.ECharts, labels: Leduc_Train_Label[], series: Leduc_Train_Serie[]) => {
    assert_chart_data(labels, series);
    chart.setOption({
        xAxis: { data: labels },
        series: series.map(x => ({ ...x, type: "line" }))
    })
}

@customElement('training-chart')
class Training_Chart extends LitElement {
    static styles = css`
        :host {
            display: block;
            height: 100%;
            width: 100%;
        }
    `

    labels: Leduc_Train_Label[] = []
    series: Leduc_Train_Serie[] = []

    chart?: echarts.ECharts;

    add_item(labels: Leduc_Train_Label[], series: Leduc_Train_Serie[]) {
        assert_chart_data(labels, series);
        if (
            this.series.length === series.length
            && this.series.every((x, i) => x.name === series[i].name)
        ) {
            this.labels.push(...labels);
            this.series.forEach((x, i) => x.data.push(...series[i].data))
        } else {
            this.labels = labels;
            this.series = series;
        }
        this.update_chart();
    }

    render() {
        // The slot will put the chart which is in the light DOM into the shadow root.
        return html`<slot></slot>`;
    }

    update_chart() {
        if (this.chart) {
            update_chart(this.chart, this.labels, this.series);
        }
    }

    firstUpdated() {
        // Create the echarts instance
        this.chart = echarts.init(this);

        // Draw the chart
        this.chart.setOption({
            title: {
                text: 'RL Agent Training Run'
            },
            yAxis: {
                boundaryGap: [0, '50%'],
                type: 'value'
            },
            tooltip: {},
            xAxis: {
                data: []
            },
        });

        // Without this the chart sizing wouldn't work properly.
        const chart_container_resize_observer = new ResizeObserver(() => this.chart?.resize());
        chart_container_resize_observer.observe(this)
    }
}
