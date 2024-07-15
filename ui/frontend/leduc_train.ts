import { LitElement, PropertyValueMap, PropertyValues, css, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { map } from 'lit/directives/map.js';
import { createRef, Ref, ref } from 'lit/directives/ref.js';
import { io } from 'socket.io-client'
import * as echarts from 'echarts';

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

const update_chart = (chart: echarts.ECharts, labels: Leduc_Train_Label[], series: Leduc_Train_Serie[]) => {
    if (series.every(x => labels.length === x.data.length) === false) { throw Error("The data sizes are not valid.") }
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
        { name: "Agent 1", data: [] },
        { name: "Agent 2", data: [] },
        { name: "Agent 3", data: [] },
    ]

    chart?: echarts.ECharts;

    render() {
        // The slot will put the chart which is in the light DOM into the shadow root.
        return html`
            <div>
                <slot></slot>
                <sl-button @click=${this.add_data}>Add Data</sl-button>
            </div>
            `;
    }

    add_data() {
        const now = new Date();
        this.labels.push(now.getSeconds().toString())
        this.series.forEach(({ data }) => {
            data.push((Math.random() - 0.4) * 10 + (data[data.length - 1] ?? 0));
        })
        if (this.chart) {
            update_chart(this.chart, this.labels, this.series)
        }
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
