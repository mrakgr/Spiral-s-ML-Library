import { LitElement, PropertyValueMap, PropertyValues, css, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { map } from 'lit/directives/map.js';
import { io } from 'socket.io-client'
import * as echarts from 'echarts';

type UI_State = {

}

@customElement('leduc-train-ui')
class Leduc_Train_UI extends LitElement {
    @property({type: Object}) state : UI_State = {
        // pl_type: players,
        // ui_game_state: ["GameNotStarted", []],
        // messages: []
    };

    // socket = io('/leduc_game')

    constructor(){
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
            display: flex;
            flex-direction: row;
            box-sizing: border-box;
            height: 100%;
        }
    `

    render(){
        return html`<canvas id="chart"></canvas>`
    }

    protected firstUpdated(_changedProperties: PropertyValues): void {
        // Create the echarts instance
        const myChart = echarts.init(document.getElementById('main'));

        // Draw the chart
        myChart.setOption({
            title: {
                text: 'ECharts Getting Started Example'
            },
            tooltip: {},
            xAxis: {
                data: ['shirt', 'cardigan', 'chiffon', 'pants', 'heels', 'socks']
            },
            yAxis: {},
            series: [
                {
                name: 'sales',
                type: 'bar',
                data: [5, 20, 36, 10, 10, 20]
                }
            ]
        });
    }
}
