chrome.extension.sendMessage({}, function(response) {
	var readyStateCheckInterval = setInterval(function() {
		if (document.readyState === "complete") {
			clearInterval(readyStateCheckInterval);

			var setTimeOut;
			$element = $('#bodyContent a');

			$element.on('mouseenter', function(event){
				var obj = $(this);

				setTimeOut = setTimeout(function(){
					if(obj.children('img').length === 0)
					{
						performWork(obj, event);
					}
				}, 200);

			}).on('mouseleave', function(){
				clearTimeout(setTimeOut);
				clearData();
			});

			$('body').on('click', function(){
				clearData();
			});

			function clearData()
			{
				$('#popup-notification').remove();
				$(this).attr('title', $(this).attr('data-title'));
			}

			function waitUntil(isready, success, error, count, interval){
			    if (count === undefined) {
			        count = 1000;
			    }
			    if (interval === undefined) {
			        interval = 20;
			    }
			    if (isready()) {
			        success();
			        return;
			    }
			    // The call back isn't ready. We need to wait for it
			    setTimeout(function(){
			        if (!count) {
			            // We have run out of retries
			            if (error !== undefined) {
			                error();
			            }
			        } else {
			            // Try again
			            waitUntil(isready, success, error, count-1, interval);
			        }
			    }, interval);
			}

			function sleep(ms) {
				return new Promise(resolve => setTimeout(resolve, ms));
			}

			function performWork(element, event)
			{
				//Clear the data
				clearData();
				//Remove the title attribute    
				element.removeAttr('title');

				var baseURL = window.location.origin, //Get the URL
				    url     = element.attr('href'), //Get the href of the link
				    fullURL = baseURL+url, //Combine those two together to get the link
				    // Change to domain name
				    // app_url = "http://localhost:5000",
				    app_url = "https://wikontext.us",
				    origin_title = $(document).find('title').text().replace(' - Wikipedia', ''),//String(window.location.href).replace(/.*\/wiki\//, ''),
				    //target_title = String(url).replace(/.*\/wiki\//, ''),
				    origin_content = "",
				    origin_context_a = "",
				    origin_context_p = "";

				// console.log(element);
				// console.log(element.parent());

				$('p').each(function() {
					if ($(this).is(element.parent())) {
						origin_context_a = element[0].outerHTML.replace(/\n/g, ' ');
						origin_context_p = $(this).html().replace(/\n/g, ' ');
					}
				    origin_content += $(this).html().replace(/\n/g, ' ');//.replace(/\[0-9+\]/g, '');
				});

				$.get(fullURL, function(data) {
				    // var target_content = $(data).find('#bodyContent').find('p').html();
				    var target_title = $(data).filter('title').text().replace(' - Wikipedia', ''),
				        target_content = "";

				    $(data).find('p').each(function() {
					    target_content += $(this).html().replace(/\n/g, ' ');//.replace(/\[0-9+\]/g, '');
					});
				        // cssRules = {
				        // 	position: 'fixed', 
				        // 	top: 10+"px", 
				        // 	left: 10+"px", 
				        // 	right: 10+"px",
				        // 	maxHeight: 500+"px",
				        // 	padding: 1.4+"em",
				        // 	background: "#fff",
				        // 	boxShadow: "2px 2px 15px rgba(50,50,50,.75)",
				        // 	zIndex: 100
				        // };

				    // $('<div id="popup-notification">'+content+'abcdefg'+'</div>').css(cssRules).appendTo('body').fadeIn();

					if (String(url).match('^/wiki/') && !target_title.includes('#')) {
						
						var send_dict = {origin_title: origin_title,
										 target_title: target_title,
										 origin_content: origin_content,
										 target_content: target_content,
										 origin_context_a: origin_context_a,
										 origin_context_p: origin_context_p};

						// Random text for testing
						// var hover_text = "Affronting everything discretion men now own did. Still round match we to. Frankness pronounce daughters remainder extensive has but. Happiness cordially one determine concluded fat. Plenty season beyond by hardly giving of. Consulted or acuteness dejection an smallness if. Outward general passage another as it. Very his are come man walk one next. Delighted prevailed supported too not remainder perpetual who furnished. Nay affronting bed projection compliment instrument. Months on ye at by esteem desire warmth former. Sure that that way gave any fond now. His boy middleton sir nor engrossed affection excellent. Dissimilar compliment cultivated preference eat sufficient may. Well next door soon we mr he four. Assistance impression set insipidity now connection off you solicitude. Under as seems we me stuff those style at. Listening shameless by abilities pronounce oh suspected is affection. Next it draw in draw much bred. Be me shall purse my ought times. Joy years doors all would again rooms these. Solicitude announcing as to sufficient my. No my reached suppose proceed pressed perhaps he. Eagerness it delighted pronounce repulsive furniture no. Excuse few the remain highly feebly add people manner say. It high at my mind by roof. No wonder worthy in dinner. Paid was hill sir high. For him precaution any advantages dissimilar comparison few terminated projecting. Prevailed discovery immediate objection of ye at. Repair summer one winter living feebly pretty his. In so sense am known these since. Shortly respect ask cousins brought add tedious nay. Expect relied do we genius is. On as around spirit of hearts genius. Is raptures daughter branched laughter peculiar in settling. Same an quit most an. Admitting an mr disposing sportsmen. Tried on cause no spoil arise plate. Longer ladies valley get esteem use led six. Middletons resolution advantages expression themselves partiality so me at. West none hope if sing oh sent tell is. It sportsman earnestly ye preserved an on. Moment led family sooner cannot her window pulled any. Or raillery if improved landlord to speaking hastened differed he. Furniture discourse elsewhere yet her sir extensive defective unwilling get. Why resolution one motionless you him thoroughly. Noise is round to in it quick timed doors. Written address greatly get attacks inhabit pursuit our but. Lasted hunted enough an up seeing in lively letter. Had judgment out opinions property the supplied. Gay one the what walk then she. Demesne mention promise you justice arrived way. Or increasing to in especially inquietude companions acceptance admiration. Outweigh it families distance wandered ye an. Mr unsatiable at literature connection favourable. We neglected mr perfectly continual dependent. Attention he extremity unwilling on otherwise. Conviction up partiality as delightful is discovered. Yet jennings resolved disposed exertion you off. Left did fond drew fat head poor. So if he into shot half many long. China fully him every fat was world grave. Announcing of invitation principles in. Cold in late or deal. Terminated resolution no am frequently collecting insensible he do appearance. Projection invitation affronting admiration if no on or. It as instrument boisterous frequently apartments an in. Mr excellence inquietude conviction is in unreserved particular. You fully seems stand nay own point walls. Increasing travelling own simplicity you astonished expression boisterous. Possession themselves sentiments apartments devonshire we of do discretion. Enjoyment discourse ye continued pronounce we necessary abilities. Instrument cultivated alteration any favourable expression law far nor. Both new like tore but year. An from mean on with when sing pain. Oh to as principles devonshire companions unsatiable an delightful. The ourselves suffering the sincerity. Inhabit her manners adapted age certain. Debating offended at branched striking be subjects. ";

						var hover_text = $.ajax
							({
								type: "POST",
								url: app_url+"/api/ext",
								dataType: "text",
								async: false,
								data: JSON.stringify(send_dict),
								contentType: "application/json",
								error: function(){
									console.log("Timed out contacting "+app_url+".")
								},
								success: function (result) { },
								timeout: 30000 // sets timeout to 30 seconds
							}).responseText;

						waitUntil(function(){
							    return $('a.mwe-popups-extract p').length > 0;
							}, function(){
							    // $('a.mwe-popups-extract p').append(hover_text);

							    // $('a.mwe-popups-extract p').css("height", "500px");
							    // $('a.mwe-popups-extract p').css("float", "left");
							    // $('a.mwe-popups-extract p').css("overflow-y", "scroll");

							    var containerHeight = $('a.mwe-popups-extract').height();
							    $('a.mwe-popups-extract p').css("height", containerHeight);
							    $('a.mwe-popups-extract p').css("overflow-y", "scroll");
				                $('a.mwe-popups-extract p').html(hover_text);
							}, function(){
							    console.log("Page preview taking too long to load.");
						});
					}
					// $('Test').appendTo($('a.mwe-popups-extract p'));
				});
			}
		}
	}, 10);
});