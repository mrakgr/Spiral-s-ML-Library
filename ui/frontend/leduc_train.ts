import { LitElement, PropertyValueMap, PropertyValues, css, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { map } from 'lit/directives/map.js';
import { createRef, Ref, ref } from 'lit/directives/ref.js';
import { io } from 'socket.io-client'
import * as echarts from 'echarts';
import { assert } from './utils';

type UI_State = {

}

@customElement('leduc-train-ui')
class Leduc_Train_UI extends LitElement {
    @property({ type: Object }) state: UI_State = {
        // pl_type: players,
        // ui_game_state: ["GameNotStarted", []],
        // messages: []
    };

    // socket = io('/leduc_game')

    constructor() {
        super()
        // this.socket.on('update', (state : UI_State) => {
        //     this.state = state;
        // });
        // this.addEventListener('game', (ev) => {
        //     ev.stopPropagation();
        //     console.log(ev);
        //     this.socket.emit('update', (ev as CustomEvent<Game_Events>).detail);
        // })
    }

    static styles = css`
        :host {
            padding: 50px;
            display: flex;
            flex-direction: row;
            box-sizing: border-box;
            height: 100%;
            width: 100%;
            /* background-color: red; */
        }

        training-chart {
            height: 60%;
            width: 60%;
        }
    `

    render() {
        return html`<training-chart></training-chart>`
    }
}

type Leduc_Train_Label = string

type Leduc_Train_Serie = {
    name: string
    data: number[]
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
    series: Leduc_Train_Serie[] = [
        {
            name: "Agent 1",
            data: []
        },
        {
            name: "Agent 2",
            data: []
        },
        {
            name: "Agent 3",
            data: []
        },
    ]

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
        return html`
            <div>
                <slot></slot>
                <sl-button @click=${this.add_test_data}>Add Data</sl-button>
            </div>
            `;
    }

    update_chart() {
        if (this.chart) {
            update_chart(this.chart, this.labels, this.series);
        }
    }


    add_test_data() {
        const now = new Date();
        this.add_item(
            [now.getSeconds().toString()],
            this.series.map(x => ({
                name: x.name,
                data: [(Math.random() - 0.4) * 10 + (x.data[x.data.length-1] ?? 0)],
            }))
        )
        this.update_chart();
    }


    firstUpdated(): void {
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
